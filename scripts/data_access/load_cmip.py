"""Load CMIP6 and CMIP7 sea ice output from the Pangeo cloud catalogs and ESGF.

The CMIP6 and CMIP7 classes search the catalogs, download/open the files, and optionally
regrid them or reduce them to sector means/sums. Derived variables (freeboard, ice density,
...) are registered on `dvr` and computed from their inputs after loading.
"""
#make a thickness threshold variable at the beginning (maybe set it to 10)
#e.g., thick_thresh=10 #m
import os
import random
import re
import socket
import time
import traceback
import warnings
from collections import ChainMap, defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Callable, Dict, Optional
from urllib.parse import urlparse

import cf_xarray as cfxr
import cftime
import intake
import intake_esgf
import numpy as np
import pandas as pd
import requests
import xarray as xr
from dask import delayed
from globus_sdk.services.search.errors import SearchAPIError
from intake_esgf import ESGFCatalog
from intake_esgf.exceptions import NoSearchResults
from intake_esm import DerivedVariableRegistry
from pyproj import Geod
from scipy.interpolate import griddata
from tqdm.auto import tqdm
from xmip.preprocessing import (broadcast_lonlat, correct_coordinates, correct_lon, fix_metadata,
                                maybe_convert_bounds_to_vertex, maybe_convert_vertex_to_bounds,
                                promote_empty_dims, rename_cmip6, sort_vertex_order)

from scripts.data_access.ancillary import NH_seaice_regions, SH_seaice_regions, grid_CESM2
from scripts.preprocessing.preprocessing import region_mask
from scripts.preprocessing.regrid import ATMOS_VARS, UNSTRUCTURED_SIDS, RegridWarning, add_corners, regrid

warnings.filterwarnings('ignore')
warnings.filterwarnings('always', category=RegridWarning)  # keep regrid()'s missing mask/areacello warnings visible

# intake-esgf's requests session has no HTTP timeout, and ESGFCatalog.search()
# blocks until every enabled index responds. A single unresponsive index (common
# with the legacy Solr nodes below) would otherwise hang search() forever.
socket.setdefaulttimeout(30)

# If a model/variable isn't found via the default Globus catalog alone, try
# uncommenting this to also search the (flakier) legacy Solr indices below —
# now safe to leave on since socket.setdefaulttimeout(30) above prevents a dead
# index from hanging search() forever.
##intake_esgf.conf.set(all_indices=True)
#intake_esgf.conf.set(indices={"esgf.nci.org.au":False})
#intake_esgf.conf.set(indices={"esg-dn1.nsc.liu.se":False})
#intake_esgf.conf.set(indices={"esgf.ceda.ac.uk":False})
#intake_esgf.conf.set(indices={"esgf-data.dkrz.de":False})
#intake_esgf.conf.set(indices={"esgf-node.ipsl.upmc.fr":False})
#intake_esgf.conf.set(indices={"esgf-node.ornl.gov":True})
#intake_esgf.conf.set(indices={"esgf-node.llnl.gov":False})
#intake_esgf.conf.set(indices={"ESGF2-US-1.5-Catalog":True})
#intake_esgf.conf.set(indices={"anl-dev":True})
#intake_esgf.conf.set(indices={"ornl-dev":True})
#intake_esgf.conf["break_on_error"] = False
# Rely on the default Globus catalog (ESGF2-US-1.5-Catalog) instead of forcing on
# all 5 legacy Solr indices, which are flaky and can each stall a search.
intake_esgf.conf.set(break_on_error=False)

def _griddata(arr, xi, method: str):
    """Fill NaNs in arr by scipy griddata interpolation from its finite points (helper for interpolate_na)."""
    ar1d = arr.ravel()
    valid = np.isfinite(ar1d)
    if valid.all():
        return arr
    return griddata(
        points=tuple(x[valid] for x in xi),
        values=ar1d[valid],
        xi=xi,
        method=method,
        fill_value=np.nan,
    ).reshape(arr.shape)
def interpolate_na(da, dim, method="nearest", use_coordinates=True, keep_attrs=True):
    """Fill NaNs in da across the dims in dim (e.g. ['y', 'x']) by 2-D scipy griddata interpolation."""
    # Create points only once.
    if use_coordinates:
        coords = [da.coords[d] for d in dim]
    else:
        coords = [np.arange(da.sizes[d]) for d in dim]

    xi = tuple(x.ravel() for x in np.meshgrid(*coords, indexing="ij"))
    arr = xr.apply_ufunc(
        _griddata,
        da,
        input_core_dims=[dim],
        output_core_dims=[dim],
        #output_dtypes=[da.dtype],
        dask="parallelized",
        vectorize=False,
        keep_attrs=keep_attrs,
        kwargs={"xi": xi, "method": method},
    ).transpose(*da.dims)
    return arr

# ── Derived variables ──────────────────────────────────────────────────────────
# Densities used for hydrostatic freeboard/ice density (kg/m3)
RHO_W = 1026
RHO_I = 916
RHO_SN = 330

def _fix_taiesm1_sithick(ds):
    """TaiESM1 sithick needs dividing by siconc twice (once as %, once as fraction) to match other models."""
    if ds.attrs.get('source_id') == 'TaiESM1':
        ds['sithick'] = ds['sithick']/ds['siconc']/(ds['siconc']/100)
    return ds

def _hydrostatic_freeboard(H_i, H_sn):
    """Ice freeboard (m) from ice and snow thickness assuming hydrostatic equilibrium; thicknesses >= 10 m are masked."""
    H_i = H_i.where(lambda x: np.abs(x) < 10)
    H_sn = H_sn.where(lambda x: np.abs(x) < 10)
    return H_i * ((RHO_W - RHO_I)/RHO_W) - H_sn * (RHO_SN/RHO_W)

dvr = DerivedVariableRegistry()
@dvr.register(variable='sifb_d', query={'variable_id': ['sithick','sisnthick','siconc']})
def calc_freeboard(ds):
    """Derived freeboard sifb_d from sithick and sisnthick (siconc only used for the TaiESM1 fix)."""
    ds = _fix_taiesm1_sithick(ds)
    ds['sifb_d'] = _hydrostatic_freeboard(ds.sithick, ds.sisnthick)
    return ds

@dvr.register(variable='sifb_d2', query={'variable_id': ['sivol','sisnthick','siconc']})
def calc_freeboard2(ds):
    """Derived freeboard sifb_d2, with ice thickness taken as sivol / siconc."""
    H_i = ds.sivol/(ds.siconc.where(ds.siconc>0)/100)
    ds['sifb_d2'] = _hydrostatic_freeboard(H_i, ds.sisnthick)
    return ds

@dvr.register(variable='sifb_d3', query={'variable_id': ['sithick','sisnthick']})
def calc_freeboard3(ds):
    """Derived freeboard sifb_d3 from sithick and sisnthick, without siconc (no TaiESM1 fix)."""
    ds['sifb_d3'] = _hydrostatic_freeboard(ds.sithick, ds.sisnthick)
    return ds

@dvr.register(variable='rhoi', query={'variable_id': ['sithick','sisnthick','sifb','siconc']})
def calc_rhoi(ds):
    """Derived ice density rhoi (kg/m3) from sifb, sisnthick and sithick by hydrostatic balance."""
    ds = _fix_taiesm1_sithick(ds)
    def valid(x):
        """Mask zeros and values >= 10 m."""
        return x.where(np.logical_and(x != 0, np.abs(x) < 10))
    ds['rhoi'] = RHO_W - ((RHO_W*valid(ds.sifb) + RHO_SN*valid(ds.sisnthick)) / valid(ds.sithick))
    return ds

@dvr.register(variable='rhoi2', query={'variable_id': ['sivol','simass','siconc']})
def calc_rhoi2(ds):
    """Derived ice density rhoi2 (kg/m3) as simass / sivol."""
    ds['rhoi2'] = ds['simass']/ds['sivol']
    return ds

@dvr.register(variable='sit_d', query={'variable_id': ['sivol','siconc']})
def calc_sit(ds):
    """Derived ice thickness sit_d as sivol / siconc (ice-covered-area mean)."""
    ds['sit_d'] = ds.sivol/(ds.siconc.where(ds.siconc>0)/100)
    return ds

def sanitize_time(ds):
    """
    Repair time coordinate for CMIP-style datasets.
    Works with cftime and datetime64.
    Handles monthly and daily data.
    """

    ds = ds.copy()

    if "time" not in ds.coords:
        return ds

    t = ds.time.values

    try:
        # If cftime objects, rebuild timestamps
        if isinstance(t[0], cftime.datetime):
            new_time = pd.to_datetime(
                [f"{tt.year:04d}-{tt.month:02d}-{tt.day:02d}" for tt in t]
            )
        else:
            new_time = pd.to_datetime(t)

        ds = ds.assign_coords(time=("time", new_time))

    except Exception as e:
        print(f"[sanitize_time] ⚠️ could not convert time: {e}")
        return ds

    # Sort time
    ds = ds.sortby("time")

    # Remove duplicate timestamps
    _, idx = np.unique(ds.time.values, return_index=True)
    ds = ds.isel(time=np.sort(idx))

    return ds

def convert_time(ds):
    """Clean the time coordinate (sanitize_time) and clip it to the experiment's standard period:
    historical 1850-2014, hist-1950 1950-2014, any other experiment 2015-2100. Returns ds unchanged
    on error.
    """
    try:
        exp_id = ds.attrs.get("intake_esm_attrs:experiment_id") or ds.attrs.get("experiment_id")

        if "time" in ds.coords:
            ds = sanitize_time(ds)

            if exp_id == "historical":
                ds = ds.sel(time=slice("1850-01-01", "2014-12-31"))
            elif exp_id == "hist-1950":
                ds = ds.sel(time=slice("1950-01-01", "2014-12-31"))
            elif exp_id:
                ds = ds.sel(time=slice("2015-01-01", "2100-12-31"))

        return ds

    except Exception as e:
        print(f"[convert_time] ⚠️ Error: {e}")
        print(ds)
        print(ds.time)
        return ds


def update_member_id(ds):
    """Standardize ensemble dims: member_id becomes '<source_id>_<variant_label>' and an experiment_id
    dimension is added.
    """
    if 'variant_label' in ds.variables:
        ds=ds.rename({'variant_label':'member_id'})
    elif 'variant_label' in ds.attrs and 'member_id' not in ds.variables:
        ds=ds.expand_dims({'member_id':[ds.attrs['variant_label']]})
    if 'sub_experiment_id' in ds.variables:
        ds = ds.isel(sub_experiment_id=0).drop_vars('sub_experiment_id')
    ds['member_id'] = (ds.attrs['source_id']+'_') + ds.member_id.astype('object')
    ds['member_id'] = ds.member_id.astype('<U25')
    ds = ds.expand_dims({'experiment_id':[ds.attrs['experiment_id']]})
    ds['experiment_id'] = ds.experiment_id.astype('<U10')
    return ds

def set_chunks(ds,time_chunks):
    """Rechunk ds along time."""
    return ds.chunk(chunks={'time': time_chunks})

def complete_preprocessing(ds):
    """Standardize a raw CMIP dataset with xMIP (dimension/coordinate names, lon range, bounds and
    vertices) and add lon_b/lat_b cell corners on 4-vertex curvilinear grids.
    """
    #ds = ds.copy()
    if 'lat' in ds.coords and 'latitude' in ds.coords and 'y' not in ds.coords:
        ds=ds.rename({'lat':'y','lon':'x'}).drop_vars(['lon_bnds','lat_bnds'], errors="ignore")
    ds = rename_cmip6(ds)
    ds = promote_empty_dims(ds)
    ds = correct_coordinates(ds)
    if 'x' in ds.variables or 'lon' in ds.variables:
        ds = broadcast_lonlat(ds)
        ds = correct_lon(ds)
        #ds = parse_lon_lat_bounds(ds)
        ds = sort_vertex_order(ds)
    #ds = correct_units(ds)
    try:
        ds = maybe_convert_bounds_to_vertex(ds)
    except Exception as error:
        pass
    ds = maybe_convert_vertex_to_bounds(ds)
    ds = fix_metadata(ds)
    if 'vertices_latitude' in ds.variables:
        ds = ds.drop_vars(['vertices_latitude','vertices_longitude'],errors='ignore')
    if 'nvertices' in ds.variables:
        ds=ds.rename({'nvertices':'vertex'})
    if 'vertex' in ds.variables and 'y' in ds.variables:
        if ds.vertex.size == 4 and ds.y.size>1:
            if 'lon_verticies' in ds.variables:
                lon_corners = cfxr.bounds_to_vertices(ds.lon_verticies.chunk(dict(vertex=-1,y=-1,x=-1)), "vertex", order=None)
                lat_corners = cfxr.bounds_to_vertices(ds.lat_verticies.chunk(dict(vertex=-1,y=-1,x=-1)), "vertex", order=None)
            if 'lon_bounds' in ds.variables and 'lon_verticies' not in ds.variables:
                lon_corners = cfxr.bounds_to_vertices(ds.lon_bounds.chunk(dict(vertex=-1,y=-1,x=-1)), "vertex", order=None)
                lat_corners = cfxr.bounds_to_vertices(ds.lat_bounds.chunk(dict(vertex=-1,y=-1,x=-1)), "vertex", order=None)
            ds=ds.assign_coords(lon_b=lon_corners, lat_b=lat_corners)
            ds=ds.rename({'y_vertices':'y_b','x_vertices':'x_b'})
            
    return ds

#load data from opendap url, needed to be in a function for list comprehension to work with try/except
def open_ds(file,chunks):
    """Open a catalog file record via its OPeNDAP URL and run complete_preprocessing; returns None on
    failure.
    """
    try: 
        ds = xr.open_dataset(file.opendap_url, chunks={'time': chunks})
        ds = complete_preprocessing(ds)
        return ds
    except Exception as error:
        pass

def calc_areacello_gufunc(lons,lats,lons2x,lats2x,lons2y,lats2y):
    """Approximate cell areas (m2) as the WGS84 geodesic east-west distance times north-south distance
    to the neighbouring points.
    """
    geod = Geod(ellps='WGS84')
    _,_, distEW = geod.inv(lons,lats,lons2x,lats2x)
    _,_, distNS = geod.inv(lons,lats,lons2y,lats2y)
    pixel_area = distEW * distNS
    return pixel_area

def calc_areacello(ds):
    """Approximate areacello from a curvilinear grid's lon/lat, for models whose areacello is missing
    or unusable.

    The returned DataArray also carries itself as an 'areacello' coordinate, so area.areacello works
    as it does for a loaded areacello Dataset.
    """
    lons=ds.lon
    lons2x=ds.lon.shift(x=1)
    lons2y=ds.lon.shift(y=1)
    lats=ds.lat
    lats2x=ds.lat.shift(x=1)
    lats2y=ds.lat.shift(y=1)
    area = xr.apply_ufunc(
        calc_areacello_gufunc,
        lons,lats,lons2x,lats2x,lons2y,lats2y,
        dask='allowed',
        output_dtypes=[float])
    area = area.fillna(area.min())
    #ds['areacello']=area
    #ds = ds.assign_coords({'areacello':area})
    area['areacello']=area
    return area

def require_all(sub_df,exp,var):
    """
    This function checks if every (`source_id`, `member_id`) combination
    has data for ALL required (`experiment_id`, `variable_id`) pairs.
    """
    if type(exp)==str:
        exp = [exp]
    if type(var)==str:
        var = [var]
    # Check if all required experiment-variable combinations exist for this `source_id`, `member_id`
    for e in exp:
        for v in var:
            if not ((sub_df["experiment_id"] == e) & (sub_df["variable_id"] == v)).any():
                return False  # Missing required data → Remove this `source_id`, `member_id`
    
    return True  # Keep this `source_id`, `member_id`

def filter_missing(sub_df,missing):
    """
    This function keeps only rows where (source_id, member_id, experiment_id)
    exist in df_missing.
    """
    merged = sub_df.merge(missing, on=["source_id", "member_id", "experiment_id"], how="inner")
    return not merged.empty


def extract_time_range_from_url(url):
    """Parse (start, end) datetimes from the YYYYMMDD-YYYYMMDD or YYYYMM-YYYYMM range in a CMIP file
    name; (None, None) if there is none.
    """
    # Try 8-digit dates first (YYYYMMDD)
    match = re.search(r"(\d{8})-(\d{8})", url)
    if match:
        start = datetime.strptime(match.group(1), "%Y%m%d")
        end = datetime.strptime(match.group(2), "%Y%m%d")
        return start, end

    # Then try 6-digit dates (YYYYMM)
    match = re.search(r"(\d{6})-(\d{6})", url)
    if match:
        start = datetime.strptime(match.group(1), "%Y%m")
        end = datetime.strptime(match.group(2), "%Y%m")
        return start, end

    # Fallback if no match
    return None, None

def is_time_range_incomplete(url, experiment_id):
    """True if a file's time range doesn't cover the standard period of experiment_id (historical,
    ssp126/245/585) or can't be parsed.
    """
    start, end = extract_time_range_from_url(url)
    if not start or not end:
        return True  # Can't determine, assume incomplete
    
    if experiment_id in ["ssp126","ssp245","ssp585"]:
        expected_start = datetime(2015, 1, 16)
        expected_end = datetime(2100, 12, 1)
    elif experiment_id == "historical":
        expected_start = datetime(1850, 1, 16)
        expected_end = datetime(2014, 12, 1)
    else:
        # Add other experiment_id ranges as needed
        return False

    return start > expected_start or end < expected_end

BAD_HOSTS = ["diasjp.net", "esgf-data04.diasjp.net"]

def _norm_url(url):
    """Normalize an ESGF URL: strip '.html' and use https."""
    url = str(url).replace(".html", "")
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    return url

def _host(url):
    """Lower-case host name of a URL ('' if it can't be parsed)."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def _basename(url):
    """File name part of a URL."""
    try:
        return os.path.basename(urlparse(url).path)
    except Exception:
        return os.path.basename(url)

def _extract_urls_from_file_info(catalog, verbose=True):
    """File URLs from an intake-esgf catalog, deduplicated and without BAD_HOSTS, with downloadable
    URLs (fileServer, Globus) ahead of OPeNDAP (dodsC).
    """
    infos = catalog._get_file_info(separator="|", quiet=not verbose)
    urls = []

    for rec in infos:
        if not isinstance(rec, dict):
            continue
        for _, v in rec.items():
            if isinstance(v, str) and ("http://" in v or "https://" in v):
                urls.append(v)
            elif isinstance(v, (list, tuple)):
                for vv in v:
                    if isinstance(vv, str) and ("http://" in vv or "https://" in vv):
                        urls.append(vv)

    # clean + dedupe + drop bad hosts
    out = []
    seen = set()
    for u in urls:
        u = _norm_url(u)
        if u in seen:
            continue
        seen.add(u)
        if any(bad in _host(u) for bad in BAD_HOSTS):
            continue
        out.append(u)

    # prefer downloadable URLs (fileServer, Globus, ...) over dodsC
    out = sorted(
        out,
        key=lambda u: (
            1 if "/thredds/dodsC/" in u else 0,
            _host(u),
            u,
        )
    )
    return out

LOCAL_ESGF_CACHE = "/tmp/esgf_manual"

def _local_cache_path(url):
    """Path under LOCAL_ESGF_CACHE where the file at url is downloaded."""
    rel = urlparse(url).path.lstrip("/")
    rel = rel.replace("thredds/fileServer/", "")
    rel = rel.replace("thredds/dodsC/", "")
    rel = rel.replace("css03_data/", "")
    return os.path.join(LOCAL_ESGF_CACHE, rel)

# Set by clear_esgf_cache() below; read by the IPython display hook installed just
# below it to add a hint alongside a FileNotFoundError that names a path under
# LOCAL_ESGF_CACHE, explaining *why* the file is gone instead of leaving a bare
# "No such file or directory" traceback.
_esgf_cache_cleared_at = None

def clear_esgf_cache():
    """Delete the raw ESGF downloads under LOCAL_ESGF_CACHE (/tmp, which has limited
    space) — call once the area-integrated outputs for a model have been saved.

    Any CMIP6_* dataset built before this call is a lazy dask/xarray object that
    still points at the now-deleted local files — reusing it afterward (re-running a
    save/diagnostic cell without first re-running the CMIP6(...).load_data() cells)
    will hit a FileNotFoundError with an added hint (see _install_stale_cache_hint)
    instead of a bare traceback.
    """
    import shutil
    global _esgf_cache_cleared_at
    if os.path.isdir(LOCAL_ESGF_CACHE):
        size_gb = sum(
            os.path.getsize(os.path.join(dirpath, f))
            for dirpath, _, filenames in os.walk(LOCAL_ESGF_CACHE)
            for f in filenames
        ) / 1e9
        shutil.rmtree(LOCAL_ESGF_CACHE)
        print(f"Cleared {size_gb:.2f} GB from {LOCAL_ESGF_CACHE}")
    else:
        print(f"{LOCAL_ESGF_CACHE} does not exist — nothing to clear")
    _esgf_cache_cleared_at = datetime.now()

def _install_stale_cache_hint():
    """In a Jupyter/IPython session, print an explanatory hint alongside (not instead
    of) the normal traceback whenever a FileNotFoundError under LOCAL_ESGF_CACHE
    surfaces after clear_esgf_cache() has run. Deliberately does NOT monkeypatch
    netCDF4/xarray internals — netCDF4.Dataset is a Cython extension type, so its
    __init__ can't be reassigned in place, and even a subclass-swap risks breaking
    xarray's internal `type(manager) is netCDF4.Dataset` checks used for multi-group
    files. IPython's set_custom_exc only changes how the exception is *displayed*, so
    it can't introduce that kind of regression."""
    try:
        from IPython import get_ipython
    except ImportError:
        return
    shell = get_ipython()
    if shell is None:
        return

    def _handler(shell, etype, evalue, tb, tb_offset=None):
        """Show the normal traceback, then the stale-cache hint if the error names a file in
        LOCAL_ESGF_CACHE.
        """
        shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
        if _esgf_cache_cleared_at is not None and LOCAL_ESGF_CACHE in str(evalue):
            print(
                f"\n[hint] {LOCAL_ESGF_CACHE} was cleared by clear_esgf_cache() at "
                f"{_esgf_cache_cleared_at:%Y-%m-%d %H:%M:%S}. The CMIP6_* dataset you're "
                "touching was built before that call, so it's now stale (it lazily points "
                "at a deleted file). Re-run the CMIP6(...).load_data() cells for this "
                "model to rebuild it before reading/saving it again."
            )

    shell.set_custom_exc((FileNotFoundError,), _handler)

_install_stale_cache_hint()

def _open_local_dataset(local_path, chunks=None, engine=None, drop_variables=None, **kwargs):
    """Open a downloaded netCDF file with CF decoding and cftime dates."""
    return xr.open_dataset(
        local_path,
        engine=engine,
        drop_variables=drop_variables,
        chunks=chunks,
        decode_cf=True,
        use_cftime=True,
        **kwargs,
    )

def _is_missing_file_error(e):
    """True if `e` indicates the remote file itself doesn't exist (vs. a transient
    network/server error) — retrying a genuinely missing file just wastes time."""
    msg = str(e).lower()
    return any(s in msg for s in (
        "file not found",
        "no such file or directory",
        "404",
    ))

def _download_file_with_retries(url, local_path, n_retries=4, timeout=60, verbose=True):
    """Download url to local_path, reusing a non-empty cached copy, with exponential-backoff retries. A
    file missing on the server is not retried. Returns local_path.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # reuse good cached file
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        if verbose:
            print(f"Using cached file: {local_path}")
        return local_path

    session = requests.Session()

    for attempt in range(1, n_retries + 1):
        tmp_path = local_path + f".part{attempt}"
        try:
            #if verbose:
            #    print(f"Download attempt {attempt}/{n_retries}: {url}")

            with session.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                raise OSError("Downloaded empty file")

            os.replace(tmp_path, local_path)

            if verbose:
                print(f"Downloaded OK: {local_path}")
            return local_path

        except Exception as e:
            if verbose:
                print(f"Download failed ({attempt}/{n_retries}): {type(e).__name__}: {e}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

            if _is_missing_file_error(e):
                if verbose:
                    print("File not found on this node — not retrying, moving on.")
                raise

            if attempt < n_retries:
                sleep_s = (2 ** (attempt - 1)) + random.uniform(0, 1)
                if verbose:
                    print(f"Sleeping {sleep_s:.1f}s before retry")
                time.sleep(sleep_s)

    raise OSError(f"All download attempts failed for {url}")

def _download_and_open(url, chunks=None, engine=None, drop_variables=None, n_retries=4, verbose=True, **kwargs):
    """Open an ESGF file: download it and open the local copy, or open OPeNDAP (dodsC) URLs remotely,
    with retries.
    """
    if "/thredds/dodsC/" not in url:
        local_path = _local_cache_path(url)
        local_path = _download_file_with_retries(
            url,
            local_path,
            n_retries=n_retries,
            verbose=verbose,
        )
        return _open_local_dataset(
            local_path,
            chunks=chunks,
            engine=engine,
            drop_variables=drop_variables,
            **kwargs,
        )

    # For dodsC, try a few times too
    last_err = None
    for attempt in range(1, n_retries + 1):
        try:
            if verbose:
                print(f"Remote open attempt {attempt}/{n_retries}: {url}")
            return xr.open_dataset(
                url,
                engine=engine,
                drop_variables=drop_variables,
                chunks=chunks,
                decode_cf=True,
                use_cftime=True,
                **kwargs,
            )
        except Exception as e:
            last_err = e
            if verbose:
                print(f"Remote open failed ({attempt}/{n_retries}): {type(e).__name__}: {e}")
            if _is_missing_file_error(e):
                if verbose:
                    print("File not found on this node — not retrying, moving on.")
                break
            if attempt < n_retries:
                sleep_s = (2 ** (attempt - 1)) + random.uniform(0, 1)
                time.sleep(sleep_s)

    raise last_err

def load_from_catalog(
    catalog,
    chunks: Optional[Dict[str, int]] = {'time': 200},
    prefer_opendap: bool = False,
    preprocess: Optional[Callable] = None,
    postprocess: Optional[Callable] = None,
    parallel: bool = True,
    engine: Optional[str] = None,
    drop_variables: Optional[list] = None,
    combine: str = 'by_coords',
    concat_dim: Optional[str] = None,
    combine_by_coords_kwargs: Optional[Dict] = None,
    combine_method: str = 'manual',
    max_files: Optional[int] = None,
    verbose: bool = True,
    esgf_url=None,
    n_retries: int = 4,
    end_year: Optional[int] = None,
    **kwargs
):
    """Download and open every file in an intake-esgf catalog; returns one Dataset per ensemble member
    (None if nothing loaded).

    Replicas of each file are tried in turn until one opens; files are concatenated over time per
    variable, and variables merged per member. preprocess and postprocess are applied to each file.
    end_year drops files that start after that year; max_files caps the number of URLs tried.
    prefer_opendap, parallel, combine, concat_dim, combine_by_coords_kwargs, combine_method and
    esgf_url are accepted but not used.
    """
    try:
        urls = _extract_urls_from_file_info(catalog, verbose=verbose)

        if verbose:
            print(f"Candidate ESGF URLs: {len(urls)}")

        if not urls:
            return None

        if max_files:
            urls = urls[:max_files]

        variant_pat = re.compile(r'_(r\d+i\d+p\d+f\d+)_')
        # start date as YYYYMM, from YYYYMM (mon), YYYYMMDD (day) or YYYYMMDDhhmm (sub-daily) stamps
        ym_pat = re.compile(r'_(\d{6})\d{0,6}-\d{6,12}')

        if end_year:
            def _keep(u):
                """False if the file starts after end_year."""
                m2 = ym_pat.search(u)
                return not (m2 and int(m2.group(1)[:4]) > end_year)
            urls = [u for u in urls if _keep(u)]
            if verbose:
                print(f"Candidate ESGF URLs after end_year filtering: {len(urls)}")
            if not urls:
                return None

        def get_variant(fname):
            """Variant label (e.g. r1i1p1f1) in a file name, or 'unknown'."""
            m = variant_pat.search(fname)
            return m.group(1) if m else "unknown"

        def get_variable(fname):
            """variable_id at the start of a file name."""
            return _basename(fname).split('_')[0]

        def start_ym(fname: str) -> int:
            """Start date of a file as YYYYMM (999999 if absent), for ordering files in time."""
            m = ym_pat.search(fname)
            return int(m.group(1)) if m else 999999

        grouped = defaultdict(list)
        for u in urls:
            grouped[get_variant(u)].append(u)

        datasets = []

        for variant, variant_urls in sorted(grouped.items()):
            by_var = defaultdict(list)
            for u in variant_urls:
                by_var[get_variable(u)].append(u)

            var_dsets = []

            for var, var_urls in sorted(by_var.items()):
                # group replicas by logical file name
                file_groups = defaultdict(list)
                for u in var_urls:
                    file_groups[_basename(u)].append(u)

                time_dsets = []

                for logical_file, candidates in sorted(file_groups.items(), key=lambda kv: start_ym(kv[0])):
                    ds = None
                    for url in candidates:
                        try:
                            #if verbose:
                            #    print("Trying", url)
                            ds = _download_and_open(
                                url,
                                chunks=chunks,
                                engine=engine,
                                drop_variables=drop_variables,
                                n_retries=n_retries,
                                **kwargs,
                            )
                            if preprocess:
                                ds = preprocess(ds)
                            if postprocess:
                                ds = postprocess(ds)
                            break
                        except Exception as e:
                            pass
                            #if verbose:
                            #    print(f"Failed: {type(e).__name__}: {e}")

                    if ds is not None:
                        time_dsets.append(ds)

                if not time_dsets:
                    continue

                var_cat = xr.concat(
                    time_dsets,
                    dim="time",
                    coords="minimal",
                    data_vars="minimal",
                    compat="override",
                    combine_attrs="override",
                )
                var_dsets.append(var_cat)

            if not var_dsets:
                continue

            merged = xr.merge(var_dsets, compat="override")
            datasets.append(merged)

        return datasets if datasets else None

    except Exception as e:
        if verbose:
            print(f"load_from_catalog failed: {type(e).__name__}: {e}")
        return None

def load_first_valid_entry(catalog):
    """Open the first file in the catalog that loads (for fixed fields such as areacello and sftof);
    None if none do.
    """
    try:
        catalog.remove_ensembles()
    except ValueError:
        # fx records (areacello/sftof) can carry variant labels intake-esgf can't parse;
        # we only need one valid file, so skip the ensemble reduction
        pass
    urls = _extract_urls_from_file_info(catalog, verbose=False)

    for url in urls:
        try:
            ds = _download_and_open(url)
            return ds
        except Exception:
            pass

    return None


# Unstructured-grid models (UNSTRUCTURED_SIDS) are regridded to grid_CESM2 for sector sums/means
# Models whose published areacello is unusable, so it is computed from lon/lat instead
CALC_AREA_SIDS = ['KIOST-ESM', 'CAS-ESM2-0', 'BCC-CSM2-MR', 'BCC-ESM1', 'NESM3']
# The area-weighted (awgt) path has always used NESM3's own areacello
CALC_AREA_SIDS_SECTOR_MEAN = ['KIOST-ESM', 'CAS-ESM2-0', 'BCC-CSM2-MR', 'BCC-ESM1']

class CMIP6():
    """Load CMIP6 sea ice output from the Pangeo cloud catalogs (preferred) and ESGF.

    variable is a CMIP6 variable_id, a shorthand (sic, sit, snt; sia and sivolume need sector_sum),
    or a derived variable registered on dvr (sifb_d, sit_d, rhoi, ...). The output is the gridded
    field (optionally regridded to new_grid with xESMF `method`), its area-weighted mean over
    sector_mean, or its sum over sector_sum. Sectors are 'NH', 'SH', 'Arctic', 'Antarctic',
    'Inner_Arctic', or {'Arctic' or 'Antarctic': [NSIDC-0780 region indices]}. members='first' loads
    one member per model; sic_mask drops cells with siconc at or below that fraction. load_data()
    loads; data_summary() returns the catalog search results.
    """
    def __init__(self, variable, experiment_id, compare_exp=None, source_id='all', sector_mean=None,sector_sum=None, members=None,time_chunks=None,
                 grid_label=['gn'], new_grid=None, method='conservative_normed',client=False,sic_mask=None,verbose=False,skip_sids=None,table_id=None,
                 end_year=None,
                 esgf_url='https://esgf-node.ornl.gov/esg-search',
                 cat_url="https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.json",
                 cat_url2="https://storage.googleapis.com/cmip6/cmip6-pgf-ingestion-test/catalog/catalog.json"):
        """Store the search/processing options and open the cloud (cat_url, cat_url2) and ESGF
        catalogs.
        """
        self.variable = variable
        self.experiment_id = experiment_id
        self.compare_exp = compare_exp
        self.source_id = source_id 
        self.sector_mean = sector_mean
        if self.sector_mean in ['Inner Arctic','IA']:
           self.sector_mean = 'Inner_Arctic'
        self.sector_sum = sector_sum
        if self.sector_sum in ['Inner Arctic','IA']:
           self.sector_sum = 'Inner_Arctic' 
        self.members = members
        self.grid_label = grid_label
        self.new_grid = new_grid
        self.method = method
        self.client = client
        self.sic_mask = sic_mask
        self.verbose = verbose
        self.skip_sids = skip_sids or []
        if type(self.skip_sids)==str:
            self.skip_sids = [self.skip_sids]
        self.end_year = end_year
        if type(self.grid_label)==str:
            self.grid_label = [self.grid_label]
        if type(self.experiment_id)==str:
            self.experiment_id = [self.experiment_id]
        if type(self.source_id)==str:
            self.source_id = [self.source_id]
        self.tid = table_id
        #self.load_from_cloud = load_from_cloud
        if self.variable in ['sia','sit','sit_d','sivolume','snt','sifb','sifb_d','sifb_d2','sifb_d3','sic','rhoi','rhoi2'] and self.tid is None:
            self.tid = 'SImon'
        if self.variable in ['tas','ts'] and self.tid is None:
            self.tid = 'Amon'
        if self.variable in ['tos'] and self.tid is None:
            self.tid = 'Omon'
        if time_chunks == None:
            if self.tid in ['SImon','Amon','Omon']:
                self.chunks = 200
            if self.tid=='SIday':
                self.chunks = 50
        if time_chunks != None:
            self.chunks = time_chunks

        self.col = intake.open_esm_datastore(cat_url,registry=dvr)
        self.col2 = intake.open_esm_datastore(cat_url2,registry=dvr)
        self.esgf_url = esgf_url
        self.col_esgf = ESGFCatalog()
            
    def _get_cat(self,vids,tid,grid,detailed=False):
        """Search the cloud and ESGF catalogs for the tgt/awgt/aswgt variables on one grid label.

        Members available in the cloud catalog are removed from the ESGF results, so each member is
        loaded from one place. Returns (cloud catalogs, ESGF catalogs), each {'tgt', 'awgt',
        'aswgt'}.
        """
        #we don't really need to reduce source ids at this point unless it is computationally expensive
        #we can search for where members/source_ids match after data is loaded
        if self.compare_exp==None:
            args = {'experiment_id':self.experiment_id,'table_id':tid,'grid_label':[],'source_id':[]}
        else:
            args = {'experiment_id':self.compare_exp,'table_id':tid,'grid_label':[],'source_id':[]}
        if grid!='all':
            args['grid_label'].extend([grid])
        if self.source_id!=['all']:
            args['source_id'].extend(self.source_id)
        args = {k: v for k, v in args.items() if len(v)!=0}
        if self.variable in ['sifb','siitdconc','siitdsnthick','siitdthick','rhoi']:
            c = self.col2
        else:
            c = self.col
        cat_tgt = c.search(**args,variable_id=vids['tgt'],require_all_on=['source_id','member_id']).search(experiment_id=self.experiment_id)
        cat_awgt = c.search(**args,variable_id=vids['awgt'],require_all_on=['source_id','member_id']).search(experiment_id=self.experiment_id)
        cat_aswgt = c.search(**args,variable_id=vids['aswgt'],require_all_on=['source_id','member_id']).search(experiment_id=self.experiment_id)
        #need to add siconc here because sit needs to be converted in 'TaiESM1' by dividing by siconc twice
        #but I just want to add siconc for 'TaiESM1' only (no need to load siconc for all models if sic_mask==None)
        if self.sic_mask==None and self.variable in ['sit'] and (self.source_id[0] in ['all','TaiESM1'] or 'TaiESM1' in self.source_id):
            #args['source_id']=['TaiESM1']
            vids2 = deepcopy(vids)
            [vids2[list(vids2.keys())[x]].append('siconc') for x in [0,1,2] if vids2[list(vids2.keys())[x]][0] is not None]
            cat_tgt2 = c.search(**args,variable_id=vids2['tgt'],require_all_on=['source_id','member_id']).search(experiment_id=self.experiment_id,source_id='TaiESM1')
            cat_awgt2 = c.search(**args,variable_id=vids2['awgt'],require_all_on=['source_id','member_id']).search(experiment_id=self.experiment_id,source_id='TaiESM1')
            cat_aswgt2 = c.search(**args,variable_id=vids2['aswgt'],require_all_on=['source_id','member_id']).search(experiment_id=self.experiment_id,source_id='TaiESM1')
            df_tgt = pd.concat([cat_tgt.df,cat_tgt2.df]).drop_duplicates()
            df_awgt = pd.concat([cat_awgt.df,cat_awgt2.df]).drop_duplicates()
            df_aswgt = pd.concat([cat_aswgt.df,cat_aswgt2.df]).drop_duplicates()
            cat_tgt.esmcat._df = df_tgt
            cat_awgt.esmcat._df = df_awgt
            cat_aswgt.esmcat._df = df_aswgt
        cat = {'tgt':cat_tgt,'awgt':cat_awgt,'aswgt':cat_aswgt}

        if self.variable in ['sifb_d','sifb_d2','sifb_d3','sit_d','rhoi','rhoi2']:
            if vids['tgt'][0]!=None:
                vids_tgt_esgf = dvr[self.variable].query['variable_id']
            else:
                vids_tgt_esgf = vids['tgt']
            if vids['awgt'][0]!=None:
                vids_awgt_esgf = dvr[self.variable].query['variable_id']
            else:
                vids_awgt_esgf = vids['awgt']
                
        else: 
            vids_tgt_esgf = vids['tgt']
            vids_awgt_esgf = vids['awgt']

        ##remove_incomplete(complete=require_all) has the same functionality of require_all_on=['source_id','member_id']
        ##it would take too much effort to make this work with compare_exp, so I think I will remove it as an argument–
        ##I don't make use of it anyway!
        try:
            cat_esgf_tgt = ESGFCatalog().search(**args,project='CMIP6',variable_id=vids_tgt_esgf,quiet=True).remove_incomplete(
                complete=lambda sub_df: require_all(sub_df, exp=self.experiment_id, var=vids_tgt_esgf))
        except (NoSearchResults, SearchAPIError) as e:
            cat_esgf_tgt = ESGFCatalog().search(
                experiment_id='historical',project='CMIP6',source_id='CESM2',variable_id='siconc',quiet=True).remove_incomplete(
                complete=lambda sub_df: require_all(sub_df, exp='ssp245', var=vids_tgt_esgf))
        try:
            cat_esgf_awgt = ESGFCatalog().search(**args,project='CMIP6',variable_id=vids_awgt_esgf,quiet=True).remove_incomplete(
                complete=lambda sub_df: require_all(sub_df, exp=self.experiment_id, var=vids_awgt_esgf))
        except (NoSearchResults, SearchAPIError) as e:
            cat_esgf_awgt = ESGFCatalog().search(
                experiment_id='historical',project='CMIP6',source_id='CESM2',variable_id='siconc',quiet=True).remove_incomplete(
                complete=lambda sub_df: require_all(sub_df, exp='ssp245', var=vids_tgt_esgf))
        try:
            cat_esgf_aswgt = ESGFCatalog().search(**args,project='CMIP6',variable_id=vids['aswgt'],quiet=True).remove_incomplete(
                complete=lambda sub_df: require_all(sub_df, exp=self.experiment_id, var=vids['aswgt']))
        except (NoSearchResults, SearchAPIError) as e:
            cat_esgf_aswgt = ESGFCatalog().search(
                experiment_id='historical',project='CMIP6',source_id='CESM2',variable_id='siconc',quiet=True).remove_incomplete(
                complete=lambda sub_df: require_all(sub_df, exp='ssp245', var=vids_tgt_esgf))
            
        cat_esgf = {'tgt':cat_esgf_tgt,'awgt':cat_esgf_awgt,'aswgt':cat_esgf_aswgt}

        if vids['awgt'][0]!=None:
            df = pd.concat([cat['tgt'].df,cat['awgt'].df]).drop_duplicates(
                subset=['source_id','member_id','experiment_id'],keep='first').reset_index(drop=True)
            df = df[df['variable_id']==vids['awgt'][0]]
            cat['awgt'].esmcat._df = df
            if vids['aswgt'][0]!=None:
                df1 = pd.concat([cat['tgt'].df,cat['awgt'].df,cat['aswgt'].search(variable_id='siconc').df]).drop_duplicates(
                        subset=['source_id','member_id','experiment_id'],keep='first').reset_index(drop=True)
                df2 = pd.concat([cat['tgt'].df,cat['awgt'].df,cat['aswgt'].search(variable_id=vids['aswgt'][0]).df]).drop_duplicates(
                        subset=['source_id','member_id','experiment_id'],keep='first').reset_index(drop=True)
                df1 = df1[df1['variable_id']=='siconc']
                df2 = df2[df2['variable_id']==vids['aswgt'][0]]
                cat['aswgt'].esmcat._df = pd.concat([df1,df2]).reset_index(drop=True)

        cloud_filtered = pd.concat([cat['tgt'].df,cat['awgt'].df,cat['aswgt'].df]).drop_duplicates(
                        subset=['source_id','member_id','experiment_id'],keep='first').reset_index(drop=True).drop(
            columns=['zstore', 'dcpp_init_year', 'activity_id', 'version','variable_id','institution_id'], errors='ignore')
        esgf_filtered = pd.concat([cat_esgf['tgt'].df,cat_esgf['awgt'].df,cat_esgf['aswgt'].df]).drop_duplicates(
                        subset=['source_id','member_id','experiment_id'],keep='first').reset_index(drop=True).drop(
            columns=['project','mip_era','activity_drs','id','version','variable_id','institution_id'], errors='ignore')
        esgf_tgt_filtered = cat_esgf['tgt'].df.drop_duplicates(
                        subset=['source_id','member_id','experiment_id'],keep='first').reset_index(drop=True).drop(
            columns=['project','mip_era','activity_drs','id','version','variable_id','institution_id'], errors='ignore')
        esgf_awgt_filtered = cat_esgf['awgt'].df.drop_duplicates(
                        subset=['source_id','member_id','experiment_id'],keep='first').reset_index(drop=True).drop(
            columns=['project','mip_era','activity_drs','id','version','variable_id','institution_id'], errors='ignore')
        esgf_aswgt_filtered = cat_esgf['aswgt'].df.drop_duplicates(
                        subset=['source_id','member_id','experiment_id'],keep='first').reset_index(drop=True).drop(
            columns=['project','mip_era','activity_drs','id','version','variable_id','institution_id'], errors='ignore')

        missing_tgt = esgf_filtered.merge(cloud_filtered, 
                                      on=['source_id', 'member_id', 'experiment_id', 'grid_label', 'table_id'], 
                                      how='left', 
                                      indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
        missing_awgt = missing_tgt.merge(esgf_tgt_filtered, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
        missing_aswgt = missing_awgt.merge(esgf_awgt_filtered, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
                                                                                                                               
        cat_esgf['tgt'].remove_incomplete(complete=lambda sub_df: filter_missing(sub_df, missing=missing_tgt))
        cat_esgf['awgt'].remove_incomplete(complete=lambda sub_df: filter_missing(sub_df, missing=missing_awgt))
        cat_esgf['aswgt'].remove_incomplete(complete=lambda sub_df: filter_missing(sub_df, missing=missing_aswgt))

        return cat,cat_esgf
    
    def _get_vids(self):
        """Variables to load for self.variable, by route: tgt (loaded as-is or regridded), awgt
        (area-weighted sector mean/sum) and aswgt (weighted by siconc*areacello and summed, e.g.
        sivolume from sithick). siconc is added to each route when sic_mask is set.
        """
        try:
            if self.variable=='sia' and self.sector_sum=='NH':
                vids = {'tgt':['siarean'],'awgt':['siconc'],'aswgt':[None]}
            if self.variable=='sia' and (self.sector_sum=='Inner_Arctic' or isinstance(self.sector_sum, dict)):
                vids = {'tgt':[None],'awgt':['siconc'],'aswgt':[None]}
            if self.variable=='sia' and self.sector_sum=='SH':
                vids = {'tgt':['siareas'],'awgt':['siconc'],'aswgt':[None]}

            if self.variable=='sivolume' and self.sector_sum=='NH':
                vids = {'tgt':['sivoln'],'awgt':['sivol'],'aswgt':['sithick','siconc']}
            if self.variable=='sivolume' and self.sector_sum=='SH':
                vids = {'tgt':['sivols'],'awgt':['sivol'],'aswgt':['sithick','siconc']}
            if self.variable=='sivolume' and (self.sector_sum=='Inner_Arctic' or isinstance(self.sector_sum, dict)):
                vids = {'tgt':[None],'awgt':['sivol'],'aswgt':['sithick','siconc']}
                
            if self.variable=='sic' and (isinstance(self.new_grid,xr.Dataset) or self.new_grid==None):
                vids = {'tgt':['siconc'],'awgt':[None],'aswgt':[None]}
            if self.variable=='sic' and self.sector_sum!=None:
                vids = {'tgt':[None],'awgt':['siconc'],'aswgt':[None]}

            if self.variable=='sit' and (isinstance(self.new_grid,xr.Dataset) or self.new_grid==None):
                vids = {'tgt':['sithick'],'awgt':[None],'aswgt':[None]}
            if self.variable=='sit' and self.sector_mean!=None:
                vids = {'tgt':[None],'awgt':['sithick'],'aswgt':[None]}

            if self.variable=='snt' and (isinstance(self.new_grid,xr.Dataset) or self.new_grid==None):
                vids = {'tgt':['sisnthick'],'awgt':[None],'aswgt':[None]}
            if self.variable=='snt' and self.sector_mean!=None:
                vids = {'tgt':[None],'awgt':['sisnthick'],'aswgt':[None]}

            if self.variable not in ['sia','sivolume','sic','sit','snt'] and (isinstance(self.new_grid,xr.Dataset) or self.new_grid==None):
                vids = {'tgt':[self.variable],'awgt':[None],'aswgt':[None]}
            if self.variable not in ['sia','sivolume','sic','sit','snt'] and self.sector_mean!=None:
                vids = {'tgt':[None],'awgt':[self.variable],'aswgt':[None]}
        
            if self.sic_mask!=None: #and self.variable not in ['sifb_d', 'sifb_d2', 'rhoi', 'rhoi2', 'sit_d']:
                [vids[list(vids.keys())[x]].append('siconc') for x in [0,1,2] if vids[list(vids.keys())[x]][0] is not None]         
            return vids
        except Exception as error:
            if self.verbose==True:
                print("An error occurred:", type(error).__name__, "–", error)
                print('Warning: Variable_id or other attribute error. Check attributes.')
    
    def _convert_to_dataset(self,ds):
        """Turn a loaded DataArray into a Dataset named self.variable, with a 'sector' or 'description'
        attribute; sifb_d2/sifb_d3 are renamed sifb_d and sit_d is renamed sit.
        """
        if self.sector_sum != None and not isinstance(self.sector_sum,dict):
            attrs={'sector':self.sector_sum}
        elif self.sector_mean != None and not isinstance(self.sector_mean,dict):
            attrs={'sector':self.sector_mean}
        elif isinstance(self.new_grid,xr.Dataset):
            attrs={'description':'regridded'}
        elif isinstance(self.sector_mean,dict):
            if list(self.sector_mean.keys())[0] == 'Arctic':
                df = NH_seaice_regions
            if list(self.sector_mean.keys())[0] == 'Antarctic':
                df = SH_seaice_regions
            attrs = {'sector':list(self.sector_mean.keys())[0]+[': '+', '.join(m for i,m in enumerate(
                df.Region.where((df.index).isin(list(self.sector_mean.values())[0])).dropna()))][0]}
        elif isinstance(self.sector_sum,dict):
            if list(self.sector_sum.keys())[0] == 'Arctic':
                df = NH_seaice_regions
            if list(self.sector_sum.keys())[0] == 'Antarctic':
                df = SH_seaice_regions
            attrs = {'sector':list(self.sector_sum.keys())[0]+[': '+', '.join(m for i,m in enumerate(
                df.Region.where((df.index).isin(list(self.sector_sum.values())[0])).dropna()))][0]}
        else:
            attrs={'description':'No spatial subsetting or regridding'}
        ds = ds.assign_attrs(attrs)
        ds = ds.to_dataset(name=self.variable)
        if self.variable == 'sifb_d2':
            ds = ds.rename({'sifb_d2':'sifb_d'})
        if self.variable == 'sifb_d3':
            ds = ds.rename({'sifb_d3':'sifb_d'})
        if self.variable == 'sit_d':
            ds = ds.rename({'sit_d':'sit'})
        ds = ds.assign_attrs(attrs)
        return ds
    
    def _fix_grids(self,ds,weight):
        """Put ds and weight (areacello or sftof) on matching coordinates.

        Copies lat/lon (and cell corners, when regridding) from weight where ds's are invalid, trims
        weight to ds's size, and flips y when one of them is stored north to south. Returns (ds,
        weight).
        """
        ds=ds.copy()
        if 'lat' in ds.coords and 'lat' in weight.coords and 'y' in ds.coords and 'x' in weight.coords:
            if ds.lat.max().values>90 or 'time' in ds.lat.dims or ds.lat.isnull().sum()>0:
                ds['lat'] = weight.lat
                ds['lon'] = weight.lon
            if weight.y.size != ds.y.size:
                weight = weight.isel(y=slice(0,ds.y.size)
                                 ,x=slice(0,ds.x.size))    
            if ds.isel(y=0).lat.values[0]!=weight.isel(y=0).lat.values[0]:
                if weight.lat.isel(y=0,x=0).values > 0:
                    weight = weight.reindex(y=list(reversed(weight.y)))
                    weight['lat'] = ds.lat
                    weight['lon'] = ds.lon
                    weight['y'] = ds.y
                    weight['x'] = ds.x
                if ds.lat.isel(y=0,x=0).values > 0:
                    ds = ds.reindex(y=list(reversed(weight.y)))
                    ds['lat'] = weight.lat
                    ds['lon'] = weight.lon
                    ds['y'] = weight.y
                    ds['x'] = weight.x
            if ds.y.values[0]!=weight.y.values[0]:
                weight['y'] = ds.y
                weight['x'] = ds.x
            if ds['lat'].mean('x').isnull().any().compute().item():
                ds['lat'] = weight.lat
                ds['lon'] = weight.lon
                ds['y'] = weight.y
                ds['x'] = weight.x
        if self.new_grid!=None:
            if ds.lon_b.max()>90 and 'lat_b' in weight.coords:
                ds['lat_b'] = weight.lat_b
                ds['lon_b'] = weight.lon_b
                ds['y_b'] = weight.y_b
                ds['x_b'] = weight.x_b
                ds['lat'] = weight.lat
                ds['lon'] = weight.lon
                ds['y'] = weight.y
                ds['x'] = weight.x
        return ds,weight

    def _remove_vars(self, ds):
        """Drop auxiliary dims and variables (dcpp_init_year, nodes, bounds, vertices, sector, ...),
        and singleton x/y for sector means/sums.
        """
        ds = ds.copy()
        if 'dcpp_init_year' in ds.dims:
            ds = ds.isel(dcpp_init_year=0)  
        if 'nodes' in ds.dims:
            ds=ds.isel(nodes=0)
        if self.sector_mean is not None or self.sector_sum is not None:
            if 'x' in ds.dims and ds.x.size==1:
                ds=ds.isel(x=0,y=0).drop_vars(['x','y','lat','lon'], errors="ignore")
            if 'vertex' in ds.dims:
                ds=ds.isel(vertex=0)
            if 'bnds' in ds.dims:
                ds=ds.isel(bnds=0)
        ds = ds.drop_vars(['sector','dcpp_init_year','nodes','type','time_bounds','vertex','bnds','iceband_bnds'], errors="ignore") 
        #else:
        #    ds = ds.drop_vars(['sector','dcpp_init_year','nodes','type','time_bounds'], errors="ignore") 
        return ds

    def _postprocessing(self, ds):
        """Per-file cleanup after loading: standard time period (convert_time), member_id/experiment_id
        dims, and unit fixes for hemispheric volumes reported in 1e3 km3.
        """
        ds = ds.copy()
        if 'time' in ds.variables:
            ds = convert_time(ds)
            #ds = convert_time2(ds)
            ds = update_member_id(ds)
            #if 'time' in ds.dims and all(var.chunks is None for var in ds.data_vars.values()):
            #ds = set_chunks(ds,self.chunks)
            ds = ds.drop_vars(['area','sector'],errors='ignore')
            #if 'units' in ds.data_vars.dtypes:
            try:
                if ds[next(iter(ds.data_vars.dtypes))].units =='1e3 km3':
                    if ds[next(iter(ds.data_vars.dtypes))].max() > 1000:
                        ds = ds/1000.
                    if ds[next(iter(ds.data_vars.dtypes))].max() > 1e8:
                        ds = ds/1e9
                    if 'y' in ds.dims:
                        ds = ds.max(['x','y'])
            except Exception as error:         
                pass
        return ds 
        
    def _get_area(self,sid,grid):
        """areacello for a model/grid from the cloud catalog, else ESGF (None if not found).
        UKESM1-1-LL uses UKESM1-0-LL's, and a few atmosphere-grid shortwave variables use areacella.
        """
        #no areacello for 'UKESM1-1-LL', so use 'UKESM1-0-LL' instead (same grid it appears)
        if sid == 'UKESM1-1-LL':
            sid = 'UKESM1-0-LL'
        #the only areacello for NESM3 has wrong resolution for some reason
        #CESM2 appears to have the same grid, with maybe slightly different ocean masking
        #f sid == 'NESM3':
        #    sid = 'CESM2'
        area_subset = self.col.search(experiment_id=self.experiment_id, variable_id=['areacello'],
                                 source_id=[sid],table_id=['Ofx'], grid_label=grid)
        if area_subset.df.source_id.size == 0:
            area_subset = self.col.search(variable_id=['areacello'],
                                     source_id=[sid],table_id=['Ofx'], grid_label=grid) 
        if self.variable in ['siflswutop','siflswdtop','siflswdbot'] and sid in ['ACCESS-CM2','MRI-ESM2-0','MPI-ESM1-2-LR']:
            area_subset = self.col.search(variable_id=['areacella'],
                                source_id=[sid],table_id=['fx'], grid_label=grid)
        df = area_subset.df.groupby(['source_id']).first().reset_index()
        area_subset.esmcat._df = df
        area_dict = area_subset.to_dataset_dict(aggregate=True,xarray_open_kwargs={"consolidated": True, 'decode_times':False}
                                                ,storage_options={"anon": True},preprocess=complete_preprocessing,progressbar=False)
        if bool(area_dict):
            area = area_dict[list(area_dict.keys())[0]].squeeze()
            area = area.where(area<1e35)
            area = area.drop_vars(['member_id','dcpp_init_year'], errors="ignore")
            if 'areacella' in area.data_vars:
                area = area.rename({'areacella':'areacello'})
            return area

        try:
            area_subset = ESGFCatalog().search(source_id=sid,project='CMIP6',variable_id='areacello',grid_label=grid,quiet=True)
            area = load_first_valid_entry(area_subset)
            if bool(area):
                area = complete_preprocessing(area)
                area = area.where(area<1e35)
                area = area.drop_vars(['member_id','dcpp_init_year'], errors="ignore")
                return area
        except NoSearchResults:
            pass  # skip silently if no data
    
    def _get_ocean_mask(self,sid,grid):
        """sftof (ocean %) for a model/grid from the cloud catalog, else ESGF (None if not found). A
        few atmosphere-grid shortwave variables use sftlf converted to an ocean fraction.
        """
#        if sid == 'UKESM1-1-LL':
#            sid = 'UKESM1-0-LL'
#        if sid == 'NESM3':
#            sid = 'CESM2'
        ocean_subset = self.col.search(experiment_id=self.experiment_id, variable_id=['sftof'],
                                  source_id=[sid],table_id=['Ofx'], grid_label=grid)
        if ocean_subset.df.source_id.size == 0:
            ocean_subset = self.col.search(variable_id=['sftof'],
                                      source_id=[sid],table_id=['Ofx'], grid_label=grid)
        if self.variable in ['siflswutop','siflswdtop','siflswdbot'] and sid in ['ACCESS-CM2','MRI-ESM2-0','MPI-ESM1-2-LR']:
            ocean_subset = self.col.search(variable_id=['sftlf'],
                                source_id=[sid],table_id=['fx'], grid_label=grid)
        df = ocean_subset.df.groupby(['source_id']).first().reset_index()
        ocean_subset.esmcat._df = df
        ocean_dict = ocean_subset.to_dataset_dict(aggregate=True,xarray_open_kwargs={"consolidated": True, 'decode_times':False}
                                                  ,storage_options={"anon": True},preprocess=complete_preprocessing,progressbar=False)
        if bool(ocean_dict):
            ocean = ocean_dict[list(ocean_dict.keys())[0]].squeeze()
            if 'sftlf' in ocean.data_vars:
                ocean = ocean.rename({'sftlf':'sftof'})
                ocean = (100-ocean)*100
            return ocean.drop_vars(['member_id','dcpp_init_year'], errors="ignore")

        try:
            ocean_subset = ESGFCatalog().search(source_id=sid,project='CMIP6',variable_id='sftof',grid_label=grid,quiet=True)
            ocean = load_first_valid_entry(ocean_subset)
            if bool(ocean):
                ocean = complete_preprocessing(ocean)
                return ocean.drop_vars(['member_id','dcpp_init_year'], errors="ignore")
        except NoSearchResults:
            pass  # skip silently if no data
    
    def _spatial_average(self,ds,weight=None,sic=None):
        """Mean of ds over the sector_mean region, weighted by cell area (or cos(lat) if weight is None).

        For sit/snt/sic only cells with values > 0 count toward the averaging area. With sic_mask,
        cells where sic <= sic_mask are excluded first.
        """
        ds = ds.copy()
        if 'lat' in ds.coords and 'y' in ds.dims:
            ds_subset = region_mask(ds, self.sector_mean)
            lat = region_mask(ds.lat, self.sector_mean)
            if self.sic_mask!=None:
                ds_subset = ds_subset.where(sic>self.sic_mask)
            if not isinstance(weight,xr.DataArray):
                #lat is already a 2D field, so each point is weighted without having to broadcast!
                #if data is on a recticular grid, then we can just weight is by the cosine of lat
                lat = lat.where(ds_subset.notnull())
                weight=np.cos(np.deg2rad(lat))/np.cos(np.deg2rad(lat)).mean('y')
                ds_mean = (ds_subset*weight).mean(['x','y'])
            else:
                #if data is on a curvilinear grid, we need to weight it by the grid-cell area
                #here, we just compute the average over the area where there is data or where 
                #sea ice variables are not 0
                ds_subset,weight = self._fix_grids(ds_subset,weight)
                #should we add sifb here, or are we allowing negative freeboard values?
                if self.variable in ['sit','snt','sic']: 
                    ds_mean = ((ds_subset*weight).sum(['x', 'y'],min_count=1)
                               /weight.where(np.logical_and(ds_subset.notnull(),ds_subset>0)).sum(['x','y']))
                else:
                    ds_mean = ((ds_subset*weight).sum(['x', 'y'],min_count=1)
                               /weight.where(ds_subset.notnull()).sum(['x','y']))
        return ds_mean
            
    def _weighted_sum(self,ds,weight,vid,calc=True):
        """Area-weighted sum of ds over the sector_sum region, scaled by 1e-12 (m2 -> 10^6 km2 for areas).

        siconc is converted from % to a fraction first. calc=False returns the weighted field
        (ds * weight) without summing, e.g. to use as weights for another variable.
        """
        ds=ds.copy()
        ds,weight = self._fix_grids(ds,weight)
        ds_subset = region_mask(ds, self.sector_sum)
        if vid == 'siconc':
            ds_subset = ds_subset/100
        if calc==False:
            #weight = (ds_subset*weight.fillna(0))
            weight = (ds_subset*weight)
            return weight
        if calc==True:
            #weighted_sum = (ds_subset*weight.fillna(0)).sum(['x', 'y'],min_count=1)*10**-12
            weighted_sum = (ds_subset*weight).sum(['x', 'y'],min_count=1)*10**-12
            return weighted_sum
    
    def _regrid(self, ds, sid, new_grid=None):
        """Regrid ds to new_grid (or self.new_grid) with scripts.preprocessing.regrid.regrid, using
        nearest_s2d for unstructured grids (UNSTRUCTURED_SIDS) and bilinear for tas/ts.
        """
        ds_out = new_grid if self.new_grid is None else self.new_grid
        return regrid(ds, ds_out, method=self.method, unstructured=sid in UNSTRUCTURED_SIDS,
                      atmos=self.variable in ATMOS_VARS, warn=False)

    def _get_area_or_calc(self, sid, grid, ds, calc_sids=CALC_AREA_SIDS):
        """areacello for sid/grid via _get_area, or computed from ds's lon/lat (calc_areacello)
        when it is missing, fails to load, or sid is in calc_sids."""
        if sid not in calc_sids:
            area = self._get_area(sid, grid)
            try:
                area.compute()
                return area
            except Exception:
                pass
        return calc_areacello(ds)

    def _sector_weights(self, sid, grid, ds, calc_sids=CALC_AREA_SIDS):
        """(areacello, new_grid) for sector means/sums. Unstructured-grid models are regridded to
        grid_CESM2 first, so they use its areacello; other models use their own (new_grid=None)."""
        if sid in UNSTRUCTURED_SIDS:
            return grid_CESM2.areacello, grid_CESM2
        return self._get_area_or_calc(sid, grid, ds, calc_sids).areacello, None

    @staticmethod
    def _valid_data_mask(da):
        """Fallback ocean mask when sftof is unavailable: cells with data in the first 120 time steps
        of the first member (first iceband, if any)."""
        if 'iceband' in da.coords:
            da = da.isel(iceband=0)
        return da.isel(time=slice(0,120)).mean(['time']).isel(member_id=0).notnull().squeeze(
        ).drop_vars(['experiment_id','time','member_id','dcpp_init_year','iceband'],errors="ignore")

    @staticmethod
    def _first_member_subset(subset):
        """Narrow a cloud catalog subset to one member: r1i1p1f1, else r1i1p1f2, r1i1p1f3, then r1i1p2f1."""
        for member in ['r1i1p1f1', 'r1i1p1f2', 'r1i1p1f3']:
            if subset.search(member_id=member).df.member_id.size >= 1:
                return subset.search(member_id=member)
        return subset.search(member_id='r1i1p2f1')

    def _regrid_unstructured(self, ds, sid, grid, vid, new_grid):
        """Attach the ocean mask (sftof, else _valid_data_mask of vid) to an unstructured-grid
        dataset and regrid it to new_grid."""
        try:
            ocean = self._get_ocean_mask(sid,grid)
            ds,ocean = self._fix_grids(ds,ocean)
            ds['mask']=(ocean.sftof/100)
        except (IndexError,AttributeError,ValueError):
            ds['mask'] = self._valid_data_mask(ds[vid])
        return self._regrid(ds,sid,new_grid=new_grid)

    def _load_data(self,vids,cat,cat_esgf,tid,grid):
        """Load every route in vids on one grid label, model by model (cloud catalog first, then ESGF),
        with regridding or sector means/sums applied. Returns one Dataset over member_id and
        experiment_id (None if the results can't be combined).
        """
        if self.variable in ['sifb_d','sifb_d2','sifb_d3','sit_d','rhoi','rhoi2']:
            agg=True
            xr_dict = {'coords':'minimal','data_vars':'minimal','compat':'override','combine_attrs':'override'}
        else:
            agg=False
            xr_dict = {}
        datasets = []
        #I used to skip these, but I keep them despite some regridding issues
        #skip_sids = ['BCC-CSM2-MR','BCC-ESM1','CAS-ESM2-0']
        
        if vids['tgt'][0] is not None:
            #if self.load_from_cloud == True:
            source_ids_cloud = cat['tgt'].df['source_id'].unique()
            source_ids_esgf = cat_esgf['tgt'].df['source_id'].unique()
            source_ids = np.union1d(source_ids_cloud,source_ids_esgf)
            if self.verbose==True:
                print('{} loaded from the cloud: {}'.format(vids['tgt'],source_ids_cloud.tolist()))
                print('{} loaded via OpenDap: {}'.format(vids['tgt'],source_ids_esgf.tolist()))
                
            for sid in (pbar := tqdm(source_ids,leave=False)):
                pbar.set_description(f"{self.variable}: {sid}")
            #for sid in source_ids:
                #if sid in skip_sids:
                #    continue
                def add_tgt(ds):
                    """Process one tgt dataset (value masks, then sector mean, regridding or
                    areacello/mask attached) and append it to datasets.
                    """
                    if sid=='TaiESM1' and self.variable in ['sit']:
                        ds['sithick']=ds['sithick']/ds['siconc']/(ds['siconc']/100)                  
                    if self.variable in ['sit','snt','sifb','sifb_d','sifb_d2','sifb_d3','sit_d']:
                        ds[vids['tgt'][0]] = ds[vids['tgt'][0]].where(lambda x:np.abs(x)<10)
                    #ds = ds.drop_vars(['area','sector'],errors='ignore')
                    if self.sic_mask!=None and self.new_grid==None:
                        sic = ds['siconc']/100
                        ds = ds.where(sic>self.sic_mask)
                    if self.sector_mean is not None:
                        if 'lat' in ds.coords and 'y' in ds.dims:
                            ds_mean = self._spatial_average(ds[vids['tgt'][0]])
                            ds_mean = self._remove_vars(ds_mean)
                            #if sid=='TaiESM1' and self.variable == 'sifb_d':
                            #    ds_mean['sifb_d'] = ds_mean.sifb_d/100
                            datasets.append(ds_mean)
                        else:
                            pass
                    elif self.new_grid is not None:
                        if self.variable not in ['tas','ts']:
                            try:
                                if 'lon_bounds' in ds.coords or 'lon_b' not in ds.coords:
                                    ds = add_corners(ds.drop_vars(['lon_bounds','lat_bounds','vertex','bounds'],errors="ignore"))
                                area = self._get_area_or_calc(sid,grid,ds)
                                ds,area = self._fix_grids(ds,area)
                                ds = ds.assign_coords({'areacello':area.areacello})
                                ocean = self._get_ocean_mask(sid,grid)
                                ds,ocean = self._fix_grids(ds,ocean)
                                ds['mask']=(ocean.sftof/100)
                                #ds['x'] = np.arange(0,len(ds.x))
                                #ds['y'] = np.arange(0,len(ds.y))
                            except (IndexError,AttributeError,ValueError):
                                if self.sic_mask!=None:
                                    ds['mask'] = self._valid_data_mask(ds['siconc'])
                                else:
                                    ds['mask'] = self._valid_data_mask(ds[vids['tgt'][0]])
                        try:
                            ds_new = self._regrid(ds,sid)
                            if self.sic_mask!=None and isinstance(self.new_grid,xr.Dataset):
                                sic = ds_new['siconc']/100
                                #if sid=='UKESM1-0-LL':
                                #    ds_new = interpolate_na(ds_new, ["y", "x"], method="linear")
                                ds_new = ds_new.where(sic>self.sic_mask)
                            ds_new = self._remove_vars(ds_new)
                            datasets.append(ds_new[vids['tgt'][0]])
                        except Exception as error:
                            if self.verbose==True:
                                print("An error occurred:", type(error).__name__, "–", error)
                                print('Warning: Regridding failed for {} (grid={})'.format(sid,grid))
                    elif self.variable in ['sia','sivolume']:
                        ds = self._remove_vars(ds)
                        datasets.append(ds[vids['tgt'][0]])
                    else:
                        if self.variable not in ['tas','ts'] and sid not in UNSTRUCTURED_SIDS:
                            try:
                                area = self._get_area_or_calc(sid,grid,ds)
                                ds,area = self._fix_grids(ds,area)
                                ds = ds.assign_coords({'areacello':area.areacello})
                            except (IndexError,AttributeError,ValueError):
                                if self.verbose==True:
                                    print('Warning: areacello from {} did not load (grid={})'.format(sid,grid))
                                ds = ds.assign_attrs(areacello='not found for this model')
                            try:
                                ocean = self._get_ocean_mask(sid,grid)
                                mask = ocean.drop_vars(['member_id','dcpp_init_year'], errors="ignore").sftof
                            except (IndexError,AttributeError,ValueError):
                                if self.sic_mask!=None:
                                    mask = sic.isel(time=0,member_id=0).notnull().squeeze(
                                    ).drop_vars(['experiment_id','time','member_id','dcpp_init_year'],errors="ignore")
                                else:
                                    mask = self._valid_data_mask(ds[vids['tgt'][0]])
                            #if sid=='ACCESS-CM2' and self.variable in ['sifb']:
                            #    ds['sifb'] = ds['sifb']/(ds['siconc']/100)
                            ds = ds.assign_coords({'mask':mask})
                        if sid in UNSTRUCTURED_SIDS:
                            try:
                                ocean = self._get_ocean_mask(sid,grid)
                                ds['mask']=(ocean.sftof/100)
                            except (IndexError,AttributeError,ValueError):
                                ds['mask'] = self._valid_data_mask(ds[vids['tgt'][0]])
                            ds = self._regrid(ds,sid,grid_CESM2)
                        datasets.append(ds[vids['tgt'][0]])
    
                if sid in source_ids_cloud:
                    if sid in self.skip_sids:
                        continue
                    try:
                        subset = cat['tgt'].search(source_id=sid)
                        if self.members=='first':
                            subset = self._first_member_subset(subset)
                            agg=True
                        try:
                            ds_dict = subset.to_dataset_dict(aggregate=agg
                                                             ,xarray_open_kwargs={"consolidated": True,'decode_times': True,'use_cftime':True, 'chunks': {'time': self.chunks}}
                                                                ,storage_options={"anon": True},preprocess=complete_preprocessing,progressbar=False
                                                        ,xarray_combine_by_coords_kwargs=xr_dict)
                        except Exception as error:
                            ds_dict = subset.to_dataset_dict(aggregate=agg
                                                             ,xarray_open_kwargs={"consolidated": True,'decode_times': True,'use_cftime':False,'chunks': {'time': self.chunks}}
                                                                ,storage_options={"anon": True},preprocess=complete_preprocessing,progressbar=False
                                                        ,xarray_combine_by_coords_kwargs=xr_dict)
                        #print(ds_dict)
                        dsets = list(map(self._postprocessing,ds_dict.values()))
                        #print(dsets)
                        dsets = list(map(self._remove_vars,dsets))
                        #print(dsets)
                        #dsets = [xr.combine_nested([ds for ds in dsets if ds.experiment_id.values[0]==exp],'member_id'
                        #        ,coords='minimal',data_vars='minimal',compat='override',combine_attrs='override') 
                        #        for exp in self.experiment_id]
                        #dsets = [xr.combine_by_coords([ds for ds in dsets if ds.experiment_id.values[0]==exp]
                        #        ,coords='minimal',data_vars='minimal',compat='override',combine_attrs='override') 
                        #        for exp in self.experiment_id]
                        dsets = [xr.concat([ds for ds in dsets if ds.experiment_id.values[0]==exp]
                                ,coords='minimal',data_vars='minimal',compat='override',combine_attrs='override',dim='member_id',join='outer') 
                                for exp in self.experiment_id]
                        #print(dsets)
                        [add_tgt(ds) for ds in dsets if len(ds.data_vars)!=0]
                    
                    except Exception as error:
                        if self.verbose==True:
                            print("An error occurred:", type(error).__name__, "–", error)
                            print('Warning:{} from {} did not load (grid={})'.format(vids['tgt'],sid,grid))
                        continue
                        
                if sid in source_ids_esgf:
                    if sid in self.skip_sids:
                        continue
                    try:
                        subset = cat_esgf['tgt'].clone().search(**ChainMap({'source_id':sid}, cat_esgf['tgt'].last_search),quiet=True
                                                               ).remove_incomplete(complete=lambda sub_df: filter_missing(sub_df, missing=cat_esgf['tgt'].df))
                        if self.members=='first':
                            subset.remove_ensembles()
                    
                        #ds_dict = subset.to_dataset_dict(prefer_streaming=True, add_measures=False, quiet=True)
                        #dsets = list(map(complete_preprocessing,ds_dict.values()))
                        #dsets = list(map(self._postprocessing,dsets))
                        dsets = load_from_catalog(
                            catalog=subset,
                            chunks = {'time':self.chunks},
                            preprocess=complete_preprocessing,
                            postprocess=self._postprocessing,
                            prefer_opendap=False,
                            combine_method='manual',
                            esgf_url = self.esgf_url,
                            end_year=self.end_year
                        )
                        #dsets = list(map(self._postprocessing,dsets))
                        dsets = list(map(self._remove_vars,dsets))
                        #try:
                        #    dsets = [xr.combine_by_coords([ds for ds in dsets if ds.experiment_id.values[0]==exp]
                        #            ,coords='minimal',data_vars='minimal',compat='override',combine_attrs='override') 
                        #            for exp in self.experiment_id]
                        #except Exception as error:
                        #    dsets = [xr.combine_nested([ds for ds in dsets if ds.experiment_id.values[0]==exp],'member_id'
                        #        ,coords='minimal',data_vars='minimal',compat='override',combine_attrs='override') 
                        #        for exp in self.experiment_id]
                        dsets = [xr.concat([ds for ds in dsets if ds.experiment_id.values[0]==exp],'member_id'
                                    ,coords='minimal',data_vars='minimal',compat='override',combine_attrs='override') 
                                    for exp in self.experiment_id]
                        if self.variable in ['sifb_d','sifb_d2','sifb_d3','sit_d','rhoi','rhoi2']:
                        #    if any(var not in dsets[0].data_vars for var in vars): ###no longer needed (I think), but needs to be revised if needed
                        #        if self.verbose==True:
                        #            print('Warning: {} not loaded from {}'.format(vids['tgt'],sid))
                        #        continue
                            dsets = list(map(dvr[self.variable].func,dsets))                    
                        [add_tgt(ds) for ds in dsets if len(ds.data_vars)!=0]
                    
                    except Exception as error:
                        if self.verbose==True:
                            print("An error occurred:", type(error).__name__, "–", error)
                            print('Warning:{} from {} did not load (grid={})'.format(vids['tgt'],sid,grid))
                        continue
                
        if vids['awgt'][0] is not None:
            source_ids_cloud = cat['awgt'].df['source_id'].unique()
            source_ids_esgf = cat_esgf['awgt'].df['source_id'].unique()
            source_ids = np.union1d(source_ids_cloud,source_ids_esgf)
            if self.verbose==True:
                print('{} loaded from the cloud: {}'.format(vids['awgt'],source_ids_cloud.tolist()))
                print('{} loaded via OpenDap: {}'.format(vids['awgt'],source_ids_esgf.tolist()))  
            
            for sid in (pbar := tqdm(source_ids,leave=False)):
                pbar.set_description(f"{self.variable}: {sid}")
            #for sid in source_ids:
                #if sid in skip_sids:
                #    continue

                def add_awgt(ds):
                    """Sector mean/sum of one awgt dataset (regridding unstructured grids first),
                    appended to datasets.
                    """
                    if sid=='TaiESM1' and self.variable in ['sit']:
                        ds['sithick']=ds['sithick']/ds['siconc']/(ds['siconc']/100)
                    if self.variable in ['sit','snt','sifb','sifb_d','sifb_d2','sifb_d3','sit_d']:
                        ds[vids['awgt'][0]] = ds[vids['awgt'][0]].where(lambda x:np.abs(x)<10)
                    if sid in UNSTRUCTURED_SIDS:
                        ds = self._regrid_unstructured(ds,sid,grid,vids['awgt'][0],new_grid)
                    if self.sector_mean is not None:
                        if self.sic_mask==None:
                            sic=None
                        else:
                            sic = ds['siconc']/100
                        ds_mean = self._spatial_average(ds=ds[vids['awgt'][0]],weight=area,sic=sic)
                        ds_mean = self._remove_vars(ds_mean)
                        datasets.append(ds_mean)
                    if self.sector_sum is not None:   
                        ds_sum = self._weighted_sum(ds=ds[vids['awgt'][0]],weight=area,vid=vids['awgt'][0])
                        ds_sum = self._remove_vars(ds_sum)
                        datasets.append(ds_sum) 
                    
                if sid in source_ids_cloud:
                    if sid in self.skip_sids:
                        continue
                    try:
                        subset = cat['awgt'].search(source_id=sid)
                        if self.members=='first':
                            subset = self._first_member_subset(subset)
                            agg=True
                        try:
                            ds_dict = subset.to_dataset_dict(aggregate=agg
                                                             ,xarray_open_kwargs={"consolidated": True,'decode_times': True,'use_cftime':True,'chunks': {'time': self.chunks}}
                                                                    ,storage_options={"anon": True},preprocess=complete_preprocessing,progressbar=False
                                                        ,xarray_combine_by_coords_kwargs=xr_dict)
                        except Exception as error:
                            ds_dict = subset.to_dataset_dict(aggregate=agg
                                                             ,xarray_open_kwargs={"consolidated": True,'decode_times': True,'use_cftime':False,'chunks': {'time': self.chunks}}
                                                                ,storage_options={"anon": True},preprocess=complete_preprocessing,progressbar=False
                                                        ,xarray_combine_by_coords_kwargs=xr_dict)
                        dsets = list(map(self._postprocessing,ds_dict.values()))
                        dsets = list(map(self._remove_vars,dsets))
                        #dsets = [xr.combine_nested([ds for ds in dsets if ds.experiment_id.values[0]==exp],'member_id'
                        #    ,coords='minimal',data_vars='minimal',compat='override',combine_attrs='override') 
                        #    for exp in self.experiment_id]
                        #dsets = [xr.combine_by_coords([ds for ds in dsets if ds.experiment_id.values[0]==exp]
                        #    ,coords='minimal',data_vars='minimal',compat='override',combine_attrs='override') 
                        #    for exp in self.experiment_id]
                        dsets = [xr.concat([ds for ds in dsets if ds.experiment_id.values[0]==exp]
                            ,coords='minimal',data_vars='minimal',compat='override',combine_attrs='override',dim='member_id',join='outer') 
                            for exp in self.experiment_id] 
                        #try:
                        area, new_grid = self._sector_weights(sid,grid,dsets[0],calc_sids=CALC_AREA_SIDS_SECTOR_MEAN)
                        #except (IndexError,AttributeError,ValueError):
                        #    if self.verbose==True:
                        #        print('Warning: area from {} not found (grid={})'.format(sid,grid))
                        #    continue
                        [add_awgt(ds) for ds in dsets if len(ds.data_vars)!=0 and ('x' and 'y' in ds.coords or sid in UNSTRUCTURED_SIDS)]
                        
                    except Exception as error:
                        if self.verbose==True:
                            print("An error occurred:", type(error).__name__, "–", error)
                            print('Warning: {} from {} did not load (grid={})'.format(vids['awgt'],sid,grid))
                        continue

                if sid in source_ids_esgf:
                    if sid in self.skip_sids:
                        continue
                    try:
                        subset = cat_esgf['awgt'].clone().search(**ChainMap({'source_id':sid}, cat_esgf['awgt'].last_search),quiet=True
                                                               ).remove_incomplete(complete=lambda sub_df: filter_missing(sub_df, missing=cat_esgf['awgt'].df))
                        if self.members=='first':
                            subset.remove_ensembles()
                    
                        #ds_dict = subset.to_dataset_dict(prefer_streaming=True, add_measures=False, quiet=True)
                        #dsets = list(map(complete_preprocessing,ds_dict.values()))
                        #dsets = list(map(self._postprocessing,dsets))
                        dsets = load_from_catalog(
                            catalog=subset,
                            chunks = {'time':self.chunks},
                            preprocess=complete_preprocessing,
                            postprocess=self._postprocessing,
                            prefer_opendap=False,
                            combine_method='manual',
                            end_year=self.end_year)
                        #dsets = list(map(self._postprocessing,dsets))
                        dsets = list(map(self._remove_vars,dsets))
                        try:
                            dsets = [xr.combine_by_coords([ds for ds in dsets if ds.experiment_id.values[0]==exp]
                                    ,coords='minimal',data_vars='minimal',compat='override',combine_attrs='override')
                                    for exp in self.experiment_id]
                        except Exception as error:
                            dsets = [xr.combine_nested([ds for ds in dsets if ds.experiment_id.values[0]==exp],'member_id'
                                ,coords='minimal',data_vars='minimal',compat='override',combine_attrs='override') 
                                for exp in self.experiment_id]
                        if self.variable in ['sifb_d','sifb_d2','sifb_d3','sit_d','rhoi','rhoi2']:
                        #    if any(var not in dsets[0].data_vars for var in vars): ###no longer needed (I think), but needs to be revised if needed
                        #        if self.verbose==True:
                        #            print('Warning: {} not loaded from {}'.format(vids['awgt'],sid))
                        #        continue
                            dsets = list(map(dvr[self.variable].func,dsets))
                        area, new_grid = self._sector_weights(sid,grid,dsets[0],calc_sids=CALC_AREA_SIDS_SECTOR_MEAN)
                        [add_awgt(ds) for ds in dsets if len(ds.data_vars)!=0 and ('x' and 'y' in ds.coords or sid in UNSTRUCTURED_SIDS)]
                        
                    except Exception as error:
                        if self.verbose==True:
                            traceback.print_exc()
                            print("An error occurred:", type(error).__name__, "–", error)
                            print('Warning:{} from {} did not load (grid={})'.format(vids['awgt'],sid,grid))
                        continue    
                        
        if vids['aswgt'][0] is not None:
            source_ids_cloud = cat['aswgt'].df['source_id'].unique()
            source_ids_esgf = cat_esgf['aswgt'].df['source_id'].unique()
            source_ids = np.union1d(source_ids_cloud,source_ids_esgf)
            if self.verbose==True:
                print('{} loaded from the cloud: {}'.format(vids['aswgt'],source_ids_cloud.tolist()))
                print('{} loaded via OpenDap: {}'.format(vids['aswgt'],source_ids_esgf.tolist()))  
                
            for sid in (pbar := tqdm(source_ids,leave=False)):
                pbar.set_description(f"{self.variable}: {sid}")
                #if sid in skip_sids:
                #    continue

                def add_aswgt(ds):
                    """Sum of one aswgt variable weighted by siconc*areacello over sector_sum, appended
                    to datasets.
                    """
                    if sid=='TaiESM1':
                        ds['sithick']=ds['sithick']/ds['siconc']/(ds['siconc']/100)
                    if sid in UNSTRUCTURED_SIDS:
                        ds = self._regrid_unstructured(ds,sid,grid,vids['aswgt'][0],new_grid)
                    ds['sithick'] = ds['sithick'].where(lambda x:np.abs(x)<10)
                    sia_grid = self._weighted_sum(ds=ds['siconc'],weight=area,vid='siconc',calc=False)
                    ds_sum = self._weighted_sum(ds=ds[vids['aswgt'][0]],weight=sia_grid,vid=vids['aswgt'][0])
                    ds_sum = self._remove_vars(ds_sum)
                    datasets.append(ds_sum)

                if sid in source_ids_cloud:
                    if sid in self.skip_sids:
                        continue
                    try:
                        subset = cat['aswgt'].search(source_id=sid)
                        if self.members=='first':
                            subset = self._first_member_subset(subset)
                            agg=True
                        ds_dict = subset.to_dataset_dict(aggregate=agg
                                                         ,xarray_open_kwargs={"consolidated": True,'decode_times': True,'use_cftime':True,'chunks': {'time': self.chunks}}
                                                                    ,storage_options={"anon": True},preprocess=complete_preprocessing,progressbar=False
                                                        ,xarray_combine_by_coords_kwargs=xr_dict)
                        dsets = list(map(self._postprocessing,ds_dict.values()))
                        dsets = list(map(self._remove_vars,dsets))
                        #dsets = [xr.combine_nested([ds for ds in dsets if ds.experiment_id.values[0]==exp],'member_id'
                        #    ,coords='minimal',data_vars='minimal',compat='override',combine_attrs='override') 
                        #    for exp in self.experiment_id]
                        #dsets = [xr.combine_by_coords([ds for ds in dsets if ds.experiment_id.values[0]==exp]
                        #    ,coords='minimal',data_vars='minimal',compat='override',combine_attrs='override') 
                        #    for exp in self.experiment_id]
                        dsets = [xr.concat([ds for ds in dsets if ds.experiment_id.values[0]==exp]
                            ,coords='minimal',data_vars='minimal',compat='override',combine_attrs='override',dim='member_id',join='outer') 
                            for exp in self.experiment_id]
                        area, new_grid = self._sector_weights(sid,grid,dsets[0])
                        [add_aswgt(ds) for ds in dsets if len(ds.data_vars)!=0 and 'x' and 'y' in ds.coords]
                        
                    except Exception as error:
                        if self.verbose==True:
                            print("An error occurred:", type(error).__name__, "–", error)                 
                            print('Warning: {} from {} did not load (grid={})'.format(vids['aswgt'],sid,grid))
                        continue

                if sid in source_ids_esgf:
                    if sid in self.skip_sids:
                        continue
                    try:
                        subset = cat_esgf['aswgt'].clone().search(**ChainMap({'source_id':sid}, cat_esgf['aswgt'].last_search),quiet=True
                                                               ).remove_incomplete(complete=lambda sub_df: filter_missing(sub_df, missing=cat_esgf['aswgt'].df))
                        if self.members=='first':
                            subset.remove_ensembles()
                    
                        #ds_dict = subset.to_dataset_dict(prefer_streaming=True, add_measures=False, quiet=True)
                        #dsets = list(map(complete_preprocessing,ds_dict.values()))
                        #dsets = list(map(self._postprocessing,dsets))
                        dsets = load_from_catalog(
                            catalog=subset,
                            chunks = {'time':self.chunks},
                            preprocess=complete_preprocessing,
                            postprocess=self._postprocessing,
                            prefer_opendap=False,
                            combine_method='manual',
                            esgf_url = self.esgf_url,
                            end_year=self.end_year
                        )
                        #dsets = list(map(self._postprocessing,dsets))
                        dsets = list(map(self._remove_vars,dsets))
                        try:
                            dsets = [xr.combine_by_coords([ds for ds in dsets if ds.experiment_id.values[0]==exp]
                                    ,coords='minimal',data_vars='minimal',compat='override',combine_attrs='override') 
                                    for exp in self.experiment_id]
                        except Exception as error:
                            dsets = [xr.combine_nested([ds for ds in dsets if ds.experiment_id.values[0]==exp],'member_id'
                                    ,coords='minimal',data_vars='minimal',compat='override',combine_attrs='override') 
                                    for exp in self.experiment_id]
                        #try:
                        area, new_grid = self._sector_weights(sid,grid,dsets[0])
                        #except (IndexError,AttributeError,ValueError):
                        #    if self.verbose==True:
                        #        print('Warning: area from {} not found (grid={})'.format(sid,grid))
                        #    continue
                        [add_aswgt(ds) for ds in dsets if len(ds.data_vars)!=0 and 'x' and 'y' in ds.coords]
                        
                    except Exception as error:
                        if self.verbose==True:
                            print("An error occurred:", type(error).__name__, "–", error)
                            print('Warning:{} from {} did not load (grid={})'.format(vids['aswgt'],sid,grid))
                        continue   
                    
        try:
            dsets = list(map(self._convert_to_dataset,datasets))
            ds = xr.concat([xr.concat([ds for ds in dsets if ds.experiment_id.values[0]==exp],'member_id'
                            ,coords='minimal',data_vars='minimal',compat='override',combine_attrs='override') 
                       for exp in self.experiment_id],'experiment_id',combine_attrs='override')
            return ds
            #return dsets
        except ValueError:
            pass

    def load_data(self):
        """Load the data on every grid label and return it as one Dataset over member_id (None if
        nothing loaded).
        """
        if self.source_id is None:
            if self.verbose==True:
                print('Warning: No source_id given')
            return 
        vids,tid=self._get_vids(),self.tid
        datasets = []
        for grid in (pbar := tqdm(self.grid_label)):
            pbar.set_description("Processing data for each grid")
            cat,cat_esgf = self._get_cat(vids=vids,tid=tid,grid=grid)
            if self.client==True:
                data = delayed(self._load_data)(vids=vids,cat=cat,cat_esgf=cat_esgf,tid=tid,grid=grid)
                data = data.compute()
            else:
                data = self._load_data(vids=vids,cat=cat,cat_esgf=cat_esgf,tid=tid,grid=grid)
            if data is None:
                continue
            #if len(data.data_vars) != 0:
            #    datasets.append(data)
            if len(data) != 0:
                datasets.append(data)
        if len(datasets) == 0:
            if self.verbose==True:
                print('Warning: No data found for grid label: {}'.format(grid))
            return
        else:
            return xr.concat(datasets,'member_id').drop_duplicates('member_id')
            #return datasets
    
    def data_summary(self):
        """Catalog search results on the first grid label, without loading: (cloud catalogs, ESGF
        catalogs).
        """
        vids,tid=self._get_vids(),self.tid
        cat,cat_esgf = self._get_cat(vids=vids,tid=tid,grid=self.grid_label[0])
        return cat,cat_esgf


# Essential Model Documentation registry (browse: .../docs/grid_viewer/horizontal/)
_EMD_URL = 'https://wcrp-cmip.github.io/Essential-Model-Documentation/'
# EMD lookups, shared across CMIP7 instances:
# source_id -> native grid labels (None if the lookup failed); grid_label -> is it a single-cell region grid
_NATIVE_GRIDS = {}
_REGION_GRIDS = {}
# ESGF STAC search endpoints, for variable long names/realms/units (not in the intake-esgf search table)
_STAC_URLS = ['https://discovery.east.esgf.io/search', 'https://discovery.west.esgf.io/search']
# (variable_id, branding suffix, region) -> {'long_name', 'realm', 'units'} ({} if the lookup failed)
_VARIABLE_INFO = {}


class CMIP7(CMIP6):
    """CMIP7 loader (ESGF only), mirroring CMIP6's interface.

    Inherits CMIP6's _fix_grids/_regrid/_spatial_average/_weighted_sum/_remove_vars/
    _convert_to_dataset and overrides the catalog/loading methods. __init__ does not call
    CMIP6.__init__, so the CMIP6 cloud catalogs are never opened. Variable names are CMIP6
    variable_ids with the tgt/awgt/aswgt structure of CMIP6._get_vids; derived variables
    expand to their inputs via dvr.

    Confirmed against live ESGF search/download (2026-09-24, 2026-09-28):
    - frequency='mon'/'day'/'fx' replaces CMIP6's table_id facet (not returned as
      a df column, but functions as a search filter). realm behaves the same way.
    - region IS a search facet (2026-10-01): gridded fields are region='glb'; hemispheric
      totals (siarea, siextent, sisnmass, sivol) are published once per hemisphere as
      region='nh'/'sh' with the same variable_id, branding (tavg-u-hm-*) and grid label.
      region= picks one; the CMIP6 names (sivoln, sivols, siarean, ...) set it for you, so
      they can be looped over with the other variables. Gridded masking (sector_mean/
      sector_sum) still happens downstream, same as for CMIP6 data in this repo.
    - grid_label values are opaque per-model codes (e.g. 'g126'), not CMIP6's
      'gn'/'gr' convention. Which grids are native to a model comes from its
      Essential Model Documentation (EMD) record (see _get_native_grids).
      grid_label='native' (default) loads only native grids; None loads every
      grid (the native copy of a member is kept); a label or list (e.g. 'g126')
      loads only those. Single-cell region grids (global/NH/SH totals, e.g. g012)
      load only when a variable has nothing else or when asked for by label.
      Output carries grid_label and native_grid coordinates along member_id.
    - areacello and sftof are published and attached as 'areacello' (coord) and
      'mask' (sftof/100 when regridding, raw sftof otherwise), like CMIP6.
    - Branded variables: a variable_id can be published in several brandings
      (variable_branding_suffix = temporal-vertical-horizontal-area, e.g. tas as
      tavg-h2m-hxy-u / tminavg-h2m-hxy-u / tmaxavg-h2m-hxy-u), which are different
      variables and don't combine. branding= (exact suffix or list, an ESGF search
      facet) picks one; otherwise the time mean (tavg-*) is kept, with a warning if
      that still leaves more than one.
    - Variable names/realms/frequencies: see the notes above _get_vids.
    """
    def __init__(self, variable, experiment_id, source_id='all', sector_mean=None, sector_sum=None,
                 frequency=None, grid_label='native', new_grid=None, method='conservative_normed',
                 members=None, time_chunks=None, realm=None, branding=None, region=None, end_year=None,
                 verbose=False):
        """Store the search/processing options (see the class docstring). No catalogs are opened until
        loading.
        """
        self.variable = variable
        self.experiment_id = [experiment_id] if isinstance(experiment_id, str) else experiment_id
        self.source_id = [source_id] if isinstance(source_id, str) else source_id
        self.sector_mean = sector_mean
        if self.sector_mean in ['Inner Arctic', 'IA']:
            self.sector_mean = 'Inner_Arctic'
        self.sector_sum = sector_sum
        if self.sector_sum in ['Inner Arctic', 'IA']:
            self.sector_sum = 'Inner_Arctic'
        if self.variable in ['sia', 'sivolume'] and self.sector_sum is None:
            raise ValueError(f"CMIP7(variable={variable!r}) needs sector_sum (e.g. 'NH', 'SH')")
        self.grid_label = [grid_label] if isinstance(grid_label, str) and grid_label != 'native' else grid_label
        self.new_grid = new_grid
        self.method = method
        self.members = members
        self.realm = realm
        self.branding = branding
        self.region = region
        self.end_year = end_year
        self.sic_mask = None  # used by CMIP6._spatial_average; not supported for CMIP7 yet
        self.verbose = verbose
        self.frequency = frequency or 'mon'
        self.chunks = time_chunks or (50 if self.frequency == 'day' else 200)

    # CMIP7 variable notes (live ESGF search, project='CMIP7', 2026-09-28):
    #   CMIP6 name  CMIP7 variable_id  frequency     branding suffix  realm
    #   siconc      siconc             mon, day      tavg-u-hxy-u     seaIce
    #   sithick     sithick            mon, day      tavg-u-hxy-si    seaIce
    #   sifb        sifb               mon           tavg-u-hxy-si    seaIce
    #   sisnthick   snd                mon, day      tavg-u-hxy-sn    seaIce (realm='land' gives land snow, -lnd)
    #   sivol       sieqthick          mon           tavg-u-hxy-si    seaIce (gridded volume per area, m)
    #   sivoln/s    sivol region=nh/sh mon           tavg-u-hm-u      seaIce (also siarean/s -> siarea,
    #                                                                 siextentn/s -> siextent, sisnmassn/s -> sisnmass)
    #   simass      simass             mon           tavg-u-hxy-si    seaIce
    #   areacello   areacello          fx            ti-u-hxy-u       ocean
    #   sftof       sftof              fx            ti-u-hxy-u       ocean
    #   tas         tas                3hr/day/mon   tavg-h2m-hxy-u   atmos
    #   ts          ts                 day, mon      tavg-u-hxy-u/si
    # - realm and variable_branding_suffix both work as search facets; realm isn't returned as a df
    #   column (data_summary gets it, with long names/units, from the ESGF STAC index).
    # - CMIP7's sivol is the hemispheric total, not CMIP6's gridded sivol, so the CMIP6 name 'sivol'
    #   (also a dvr input, e.g. sifb_d2, sivolume) searches sieqthick, unless region='nh'/'sh' is set
    #   (then it is CMIP7's sivol, same as sivoln/sivols). Hemispheric totals (2026-10-01:
    #   CanESM6-0-MR only) are on the model's native grid label, not the single-cell g011/g012, and are
    #   told apart by region; sia/sivolume could load them as 'tgt' (like load.py does with siarean).
    # - Derived variables (sifb_d, sifb_d3, rhoi, ...) expand to their inputs via dvr, so new
    #   ones only need registering on dvr.
    _SHORT_NAMES = {'sic': 'siconc', 'sit': 'sithick', 'snt': 'sisnthick'}
    # CMIP6 variable_id -> CMIP7 variable_id where they differ (renamed back after loading)
    _CMIP7_NAMES = {'sisnthick': 'snd', 'sivol': 'sieqthick'}
    # CMIP6 hemispheric totals -> (CMIP7 variable_id, region)
    _HEMI_NAMES = {f'{v6}{h}': (v7, r) for v6, v7 in [('sivol', 'sivol'), ('siarea', 'siarea'),
                                                     ('siextent', 'siextent'), ('sisnmass', 'sisnmass')]
                   for h, r in [('n', 'nh'), ('s', 'sh')]}

    def _get_vids(self):
        """Variables to load by route, using CMIP6 variable_ids (same keys/semantics as
        CMIP6._get_vids): tgt is loaded as-is (or regridded to new_grid), awgt gives an
        area-weighted sector mean (sector_mean) or sum (sector_sum), and aswgt is weighted by
        siconc*areacello and summed (sivolume from sithick).
        """
        vid = self._SHORT_NAMES.get(self.variable, self.variable)
        if self.variable == 'sia':
            # CMIP7 hemispheric siarea isn't published yet, so sum siconc*areacello instead
            return {'tgt': [None], 'awgt': ['siconc'], 'aswgt': [None]}
        if self.variable == 'sivolume':
            return {'tgt': [None], 'awgt': ['sivol'], 'aswgt': ['sithick', 'siconc']}
        if self.sector_mean is not None or self.sector_sum is not None:
            return {'tgt': [None], 'awgt': [vid], 'aswgt': [None]}
        return {'tgt': [vid], 'awgt': [None], 'aswgt': [None]}

    def _search_vid(self, vid):
        """CMIP6 variable_id -> (CMIP7 variable_id, extra ESGF search facets: realm, branding, region).
        """
        facets = {}
        if self.realm is not None:
            facets['realm'] = self.realm
        elif vid == 'sisnthick' or self.variable in ['sic', 'sit', 'snt', 'sia', 'sivolume']:
            facets['realm'] = 'seaIce'
        # branding/region apply to the requested variable, not to the inputs of derived variables/sia/sivolume
        requested = vid == self._SHORT_NAMES.get(self.variable, self.variable)
        if self.branding is not None and requested:
            facets['variable_branding_suffix'] = self.branding
        if vid in self._HEMI_NAMES:
            search_vid, facets['region'] = self._HEMI_NAMES[vid]
            return search_vid, facets
        if self.region is not None and requested:
            facets['region'] = self.region
            if vid == 'sivol' and self.region != 'glb':
                return vid, facets  # a hemisphere asked for: CMIP7's sivol total, not the gridded sieqthick
        return self._CMIP7_NAMES.get(vid, vid), facets

    def _get_cat(self, vid):
        """ESGF catalog of CMIP7 datasets for one (CMIP6-named) variable, keeping members that have
        every requested experiment; None if nothing is found.
        """
        search_vid, facets = self._search_vid(vid)
        args = {
            'project': 'CMIP7',
            'variable_id': search_vid,
            'frequency': self.frequency,
            'quiet': True,
        }
        # experiment_id=None searches every experiment (for data_summary; load_data needs experiments)
        if self.experiment_id is not None:
            args['experiment_id'] = self.experiment_id
        args.update(facets)
        if self.source_id != ['all']:
            args['source_id'] = self.source_id
        if isinstance(self.grid_label, list):
            args['grid_label'] = self.grid_label
        try:
            cat = ESGFCatalog().search(**args).remove_incomplete(
                complete=lambda sub_df: require_all(sub_df, exp=self.experiment_id or [], var=search_vid))
        except (NoSearchResults, SearchAPIError) as error:
            if self.verbose:
                print(f"Warning: no CMIP7 data found for {search_vid} (grid_label={self.grid_label}): {error}")
            return None
        return cat

    def _get_native_grids(self, sid):
        """Native grid labels for a model from its EMD record: model -> model_components (e.g.
        'sea-ice_cice-gsi8_h142_v107') -> horizontal_computational_grid h142 -> horizontal_subgrids
        ('g126-mass', 'g216-velocity') -> ['g126', 'g216']. Covers every component (ocean, sea ice,
        atmosphere, ...). None if the lookup fails.
        """
        if sid not in _NATIVE_GRIDS:
            try:
                model = requests.get(f'{_EMD_URL}model/{sid.lower()}.json', timeout=30)
                model.raise_for_status()
                grids = set()
                for comp in model.json()['model_components']:
                    h = re.search(r'_(h\d+)_', comp)
                    if h is None:
                        continue
                    hgrid = requests.get(f'{_EMD_URL}horizontal_computational_grid/{h.group(1)}.json', timeout=30)
                    hgrid.raise_for_status()
                    grids.update(sub.split('-')[0] for sub in hgrid.json()['horizontal_subgrids'])
                _NATIVE_GRIDS[sid] = sorted(grids)
            except Exception as error:
                if self.verbose:
                    print(f'Warning: EMD native-grid lookup failed for {sid}: {type(error).__name__} - {error}')
                _NATIVE_GRIDS[sid] = None
        return _NATIVE_GRIDS[sid]

    def _is_region_grid(self, grid):
        """True for single-cell grids (e.g. g010 global, g011 NH, g012 SH, g190 global) that hold
        regional totals rather than gridded fields: n_cells == 1 in EMD
        horizontal_grid_cell/<grid>.json.
        """
        if grid not in _REGION_GRIDS:
            try:
                cell = requests.get(f'{_EMD_URL}horizontal_grid_cell/{grid}.json', timeout=30)
                cell.raise_for_status()
                _REGION_GRIDS[grid] = cell.json().get('n_cells') == 1
            except Exception as error:
                if self.verbose:
                    print(f'Warning: EMD grid lookup failed for {grid}: {type(error).__name__} - {error}')
                _REGION_GRIDS[grid] = False
        return _REGION_GRIDS[grid]

    def _select_grids(self, sid, grids):
        """Grids published for sid -> the ones to load, native first (see grid_label in the class
        docstring).
        """
        native = self._get_native_grids(sid) or []
        grids = sorted(grids, key=lambda g: (g not in native, g))
        if isinstance(self.grid_label, list):
            return grids  # explicit labels were already filtered in _get_cat
        gridded = [g for g in grids if not self._is_region_grid(g)]
        if not gridded:
            return grids  # only regional totals published (e.g. siarea on g011/g012)
        # region grids are skipped when gridded data exist, so a total never replaces a field
        if self.grid_label is None:
            return gridded
        keep = [g for g in gridded if g in native]
        if not keep:
            print(f'Warning: [{sid}] no native grid among published {gridded} (EMD native: {native or None}); '
                  f'loading all of them')
            return gridded
        return keep

    def _get_area(self, sid, grid):
        """CMIP7 counterpart of CMIP6._get_area (ESGF only; there is no cloud catalog)."""
        args = {'project': 'CMIP7', 'source_id': sid, 'variable_id': 'areacello', 'quiet': True}
        if grid is not None:
            args['grid_label'] = grid
        try:
            area_subset = ESGFCatalog().search(**args)
            area = load_first_valid_entry(area_subset)
            if bool(area):
                area = complete_preprocessing(area)
                area = area.where(area < 1e35)
                area = area.drop_vars(['member_id', 'dcpp_init_year'], errors="ignore")
                return area
        except (NoSearchResults, SearchAPIError):
            pass  # skip silently if no data

    def _get_ocean_mask(self, sid, grid):
        """CMIP7 counterpart of CMIP6._get_ocean_mask (ESGF only; there is no cloud catalog)."""
        args = {'project': 'CMIP7', 'source_id': sid, 'variable_id': 'sftof', 'quiet': True}
        if grid is not None:
            args['grid_label'] = grid
        try:
            ocean_subset = ESGFCatalog().search(**args)
            ocean = load_first_valid_entry(ocean_subset)
            if bool(ocean):
                ocean = complete_preprocessing(ocean)
                return ocean.drop_vars(['member_id', 'dcpp_init_year'], errors="ignore")
        except (NoSearchResults, SearchAPIError):
            pass  # skip silently if no data

    def _postprocessing(self, ds):
        """Per-file cleanup: standard time period, member_id/experiment_id dims, with full-length CMIP7
        experiment and member names.
        """
        ds = ds.copy()
        if 'time' in ds.variables:
            ds = convert_time(ds)
        ds = update_member_id(ds)
        # update_member_id casts to '<U10'/'<U25', which truncates CMIP7 names
        # (e.g. 'esm-scen7-h' -> 'esm-scen7-'); restore the full-length strings.
        ds['experiment_id'] = [ds.attrs['experiment_id']]
        if ds.sizes.get('member_id') == 1 and 'variant_label' in ds.attrs:
            ds['member_id'] = [f"{ds.attrs['source_id']}_{ds.attrs['variant_label']}"]
        return ds

    def _load_sid(self, cat, sid, vid, grid):
        """Load one variable for one source_id and grid -> {experiment_id: Dataset concatenated over
        member_id}. If several brandings or regions are published, one is chosen and a warning
        printed.
        """
        subset = cat.clone().search(**ChainMap({'source_id': sid, 'grid_label': grid}, cat.last_search), quiet=True)
        brandings = sorted(subset.df['variable_branding_suffix'].unique())
        if len(brandings) > 1:
            # several brandings are different variables (e.g. tas tavg/tminavg/tmaxavg) and don't
            # combine: keep the time mean, else the first, and say so
            tavg = [b for b in brandings if b.startswith('tavg-')]
            chosen = (tavg or brandings)[0]
            if len(tavg) != 1:
                print(f'Warning: [{sid}] {vid} is published as {brandings}; loading {chosen!r} '
                      f'(choose with branding= or realm=)')
            subset = cat.clone().search(**ChainMap({'source_id': sid, 'grid_label': grid,
                                                    'variable_branding_suffix': chosen}, cat.last_search), quiet=True)
        regions = sorted(subset.df['region'].dropna().unique()) if 'region' in subset.df else []
        if len(regions) > 1:
            # e.g. sivol is published once per hemisphere (nh/sh) with the same name, branding and grid;
            # loading both would stack them as duplicate members, so keep one and say so
            chosen = 'glb' if 'glb' in regions else regions[0]
            print(f'Warning: [{sid}] {vid} is published for regions {regions}; loading {chosen!r} '
                  f'(choose with region=, or use e.g. sivoln/sivols)')
            subset = cat.clone().search(**ChainMap({'source_id': sid, 'grid_label': grid, 'region': chosen},
                                                   subset.last_search), quiet=True)
        if self.members == 'first':
            subset.remove_ensembles()

        dsets = load_from_catalog(
            catalog=subset,
            chunks={'time': self.chunks},
            preprocess=complete_preprocessing,
            postprocess=self._postprocessing,
            prefer_opendap=False,
            combine_method='manual',
            verbose=self.verbose,
            end_year=self.end_year,
        )
        if not dsets:
            if self.verbose:
                print(f'[{sid}] load_from_catalog returned no datasets for {vid} (grid={grid})')
            return {}

        if self.verbose:
            print(f'[{sid}] load_from_catalog returned {len(dsets)} dataset(s) for {vid} (grid={grid}):')
        ##    for ds in dsets:
        ##        t = (f'{ds.time.values[0]} to {ds.time.values[-1]} (n={ds.time.size})'
        ###             if 'time' in ds.coords else 'no time coord')
        #        print(f'    experiment_id={ds.experiment_id.values.tolist()}, '
        ##              f'member_id={ds.member_id.values.tolist()}, time={t}')

        by_exp = {}
        for exp in self.experiment_id:
            matched = [ds for ds in dsets if ds.experiment_id.values[0] == exp]
            ##if self.verbose:
            ##    print(f'[{sid}] {len(matched)} dataset(s) matched experiment_id={exp!r}')
            if not matched:
                if self.verbose:
                    found = sorted({str(ds.experiment_id.values[0]) for ds in dsets})
                    print(f'Warning: [{sid}] nothing matched {exp!r}; '
                          f'experiment_ids present: {found}')
                continue
            ds = xr.concat(
                matched,
                'member_id', coords='minimal', data_vars='minimal',
                compat='override', combine_attrs='override',
            )
            # back to the CMIP6 name so dvr functions and _get_vids names work unchanged
            search_vid = self._search_vid(vid)[0]
            if search_vid != vid and search_vid in ds.data_vars:
                ds = ds.rename({search_vid: vid})
            by_exp[exp] = ds
        return by_exp

    def _load_vids(self, vids):
        """Load all variables in vids (derived names expand to their dvr inputs), merged per
        source_id/grid/experiment (members with every input only), with dvr functions applied. Grids
        are chosen per source_id by _select_grids. Returns {(source_id, grid): [Dataset per
        experiment]}.
        """
        inputs = []
        for vid in vids:
            inputs += dvr[vid].query['variable_id'] if vid in dvr else [vid]
        cats = {vid: self._get_cat(vid) for vid in inputs}
        if any(cat is None or cat.df.empty for cat in cats.values()):
            return {}

        source_ids = sorted(set.intersection(*(set(cat.df['source_id']) for cat in cats.values())))
        if self.verbose:
            print(f'{inputs} available from: {source_ids}')

        out = {}
        for sid in (pbar := tqdm(source_ids, leave=False)):
            pbar.set_description(f"{self.variable}: {sid}")
            grids = set.intersection(*(set(cat.df.loc[cat.df['source_id'] == sid, 'grid_label'])
                                       for cat in cats.values()))
            for grid in self._select_grids(sid, grids):
                try:
                    per_vid = [self._load_sid(cats[vid], sid, vid, grid) for vid in inputs]
                    exps = [exp for exp in self.experiment_id if all(exp in d for d in per_vid)]
                    dsets = [xr.merge([d[exp] for d in per_vid], join='inner', compat='override',
                                      combine_attrs='override') for exp in exps]
                    for vid in vids:
                        if vid in dvr:
                            dsets = list(map(dvr[vid].func, dsets))
                    out[(sid, grid)] = [ds for ds in dsets if len(ds.data_vars) != 0]
                except Exception as error:
                    if self.verbose:
                        print("An error occurred:", type(error).__name__, "-", error)
                        print(f'Warning: {inputs} from {sid} did not load (grid={grid})')
        return out

    def _add_area_mask(self, ds, sid, grid, vid, regrid):
        """Attach areacello and the ocean mask, as CMIP6._load_data does. With regrid=True, cell
        corners are added and the mask is sftof/100 (for xESMF); otherwise the mask is the raw
        sftof.
        """
        if self.variable in ['tas', 'ts']:
            return ds
        if regrid:
            try:
                if 'lon_bounds' in ds.coords or 'lon_b' not in ds.coords:
                    ds = add_corners(ds.drop_vars(['lon_bounds', 'lat_bounds', 'vertex', 'bounds'], errors="ignore"))
                area = self._get_area_or_calc(sid, grid, ds, calc_sids=())
                ds, area = self._fix_grids(ds, area)
                ds = ds.assign_coords({'areacello': area.areacello})
                ocean = self._get_ocean_mask(sid, grid)
                ds, ocean = self._fix_grids(ds, ocean)
                ds['mask'] = (ocean.sftof/100)
            except (IndexError, AttributeError, ValueError):
                ds['mask'] = self._valid_data_mask(ds[vid])
        else:
            try:
                area = self._get_area_or_calc(sid, grid, ds, calc_sids=())
                ds, area = self._fix_grids(ds, area)
                ds = ds.assign_coords({'areacello': area.areacello})
            except (IndexError, AttributeError, ValueError):
                if self.verbose:
                    print(f'Warning: areacello from {sid} did not load (grid={grid})')
                ds = ds.assign_attrs(areacello='not found for this model')
            try:
                ocean = self._get_ocean_mask(sid, grid)
                mask = ocean.drop_vars(['member_id', 'dcpp_init_year'], errors="ignore").sftof
            except (IndexError, AttributeError, ValueError):
                mask = self._valid_data_mask(ds[vid])
            ds = ds.assign_coords({'mask': mask})
        return ds

    def _tag_grid(self, da, sid, grid):
        """Record which grid each member came from, and whether it is native to the model, as member_id
        coordinates.
        """
        n = da.sizes['member_id']
        return da.assign_coords(grid_label=('member_id', [grid] * n),
                                native_grid=('member_id', [grid in (self._get_native_grids(sid) or [])] * n))

    def _load_data(self, vids):
        """Load every route in vids from ESGF, with regridding or sector means/sums applied. Returns a
        list of DataArrays, one per model, grid and experiment.
        """
        datasets = []

        if vids['tgt'][0] is not None:
            vid = vids['tgt'][0]
            for (sid, grid), dsets in self._load_vids(vids['tgt']).items():
                for ds in dsets:
                    try:
                        ds = self._add_area_mask(ds, sid, grid, vid, regrid=self.new_grid is not None)
                        if self.new_grid is not None:
                            ds = self._regrid(ds, sid)
                        datasets.append(self._tag_grid(self._remove_vars(ds[vid]), sid, grid))
                    except Exception as error:
                        if self.verbose:
                            print("An error occurred:", type(error).__name__, "-", error)
                            print(f'Warning: {vid} from {sid} failed after loading (grid={grid})')

        if vids['awgt'][0] is not None:
            vid = vids['awgt'][0]
            for (sid, grid), dsets in self._load_vids(vids['awgt']).items():
                for ds in dsets:
                    try:
                        if self.variable in ['sit', 'snt', 'sifb', 'sifb_d', 'sifb_d2', 'sifb_d3', 'sit_d']:
                            ds[vid] = ds[vid].where(lambda x: np.abs(x) < 10)
                        ds = self._add_area_mask(ds, sid, grid, vid, regrid=False)
                        if self.sector_mean is not None:
                            datasets.append(self._tag_grid(self._remove_vars(self._spatial_average(ds=ds[vid], weight=ds['areacello'])), sid, grid))
                        if self.sector_sum is not None:
                            datasets.append(self._tag_grid(self._remove_vars(self._weighted_sum(ds=ds[vid], weight=ds['areacello'], vid=vid)), sid, grid))
                    except Exception as error:
                        if self.verbose:
                            print("An error occurred:", type(error).__name__, "-", error)
                            print(f'Warning: sector mean/sum of {vid} from {sid} failed (grid={grid})')

        if vids['aswgt'][0] is not None:
            vid = vids['aswgt'][0]
            for (sid, grid), dsets in self._load_vids(vids['aswgt']).items():
                for ds in dsets:
                    try:
                        ds = self._add_area_mask(ds, sid, grid, vid, regrid=False)
                        ds['sithick'] = ds['sithick'].where(lambda x: np.abs(x) < 10)
                        sia_grid = self._weighted_sum(ds=ds['siconc'], weight=ds['areacello'], vid='siconc', calc=False)
                        datasets.append(self._tag_grid(self._remove_vars(self._weighted_sum(ds=ds[vid], weight=sia_grid, vid=vid)), sid, grid))
                    except Exception as error:
                        if self.verbose:
                            print("An error occurred:", type(error).__name__, "-", error)
                            print(f'Warning: siconc-weighted sum of {vid} from {sid} failed (grid={grid})')

        return datasets

    def load_data(self):
        """Load the data and return it as one Dataset over member_id; a member found more than once
        keeps its native-grid copy. None if nothing loaded.
        """
        if self.experiment_id is None:
            raise ValueError('CMIP7.load_data needs an experiment_id (None is only for data_summary)')
        if self.source_id is None:
            if self.verbose:
                print('Warning: No source_id given')
            return

        vids = self._get_vids()
        all_datasets = self._load_data(vids)

        if not all_datasets:
            if self.verbose:
                print(f'Warning: no data found for variable={self.variable}')
            return

        if self.verbose:
            n_members = sum(da.sizes.get('member_id', 1) for da in all_datasets)
            print(f'Concatenating {len(all_datasets)} dataset(s), {n_members} member(s) total')
        # a member found more than once (on several grids with grid_label=None, or by several routes,
        # e.g. sivolume from sivol and sithick) keeps its native-grid copy, then the first route
        all_datasets.sort(key=lambda da: not bool(da.native_grid.all()))
        da = xr.concat(all_datasets, 'member_id').drop_duplicates('member_id')
        return self._convert_to_dataset(da)

    def _get_variable_info(self, vid, branding, region=None):
        """Long name, realm(s) and units of a branded CMIP7 variable from the ESGF STAC index (the
        intake-esgf search table doesn't include them); {} if the lookup fails. region matters for
        hemispheric totals (e.g. sivol nh/sh: 'Sea-Ice Volume North'/'South').
        """
        key = (vid, branding, region)
        if key not in _VARIABLE_INFO:
            args = [{'op': '=', 'args': [{'property': 'properties.cmip7:variable_id'}, vid]},
                    {'op': '=', 'args': [{'property': 'properties.cmip7:variable_branding_suffix'}, branding]}]
            if region is not None:
                args.append({'op': '=', 'args': [{'property': 'properties.cmip7:region'}, region]})
            query = {'collections': ['CMIP7'], 'limit': 1, 'filter': {'op': 'and', 'args': args}}
            _VARIABLE_INFO[key] = {}
            for url in _STAC_URLS:
                try:
                    r = requests.post(url, json=query, timeout=30)
                    r.raise_for_status()
                    props = r.json()['features'][0]['properties']
                    realm = props.get('cmip7:realm')
                    _VARIABLE_INFO[key] = {'long_name': props.get('cmip7:variable_long_name'),
                                           'realm': ', '.join(realm) if isinstance(realm, list) else realm,
                                           'units': props.get('cmip7:variable_units')}
                    break
                except Exception as error:
                    if self.verbose:
                        print(f'Warning: STAC lookup at {url} failed for {vid}_{branding}: {type(error).__name__} - {error}')
        return _VARIABLE_INFO[key]

    def data_summary(self):
        """ESGF search results for each route of _get_vids (derived variables expand to their dvr
        inputs), without downloading anything.

        Returns {'tgt': df, 'awgt': df, 'aswgt': df} like CMIP6.data_summary, but DataFrames rather
        than catalogs since CMIP7 has no cloud catalog; pd.concat(summary, names=['route']) gives
        one table. One row per ESGF dataset, so a variable published in several brandings shows
        each. frequency replaces CMIP6's table_id; realm, long_name and units come from the ESGF
        STAC index; native_grid/region_grid come from EMD.
        """
        columns = ['project', 'activity_id', 'institution_id', 'source_id', 'experiment_id',
                   'variant_label', 'frequency', 'realm', 'variable_id', 'long_name', 'units',
                   'variable_branding_suffix', 'region', 'grid_label', 'version', 'id', 'native_grid', 'region_grid']
        summary = {}
        for route, group in self._get_vids().items():
            dfs = []
            if group[0] is not None:
                inputs = []
                for vid in group:
                    inputs += dvr[vid].query['variable_id'] if vid in dvr else [vid]
                for vid in inputs:
                    cat = self._get_cat(vid)
                    if cat is None or cat.df.empty:
                        continue
                    df = cat.df.copy()
                    info = [self._get_variable_info(v, b, r) for v, b, r in
                            zip(df['variable_id'], df['variable_branding_suffix'], df['region'])]
                    for col in ['long_name', 'realm', 'units']:
                        df[col] = [i.get(col) for i in info]
                    df['native_grid'] = [g in (self._get_native_grids(sid) or [])
                                         for sid, g in zip(df['source_id'], df['grid_label'])]
                    df['region_grid'] = df['grid_label'].map(self._is_region_grid)
                    dfs.append(df.reindex(columns=columns))
            summary[route] = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=columns)
        return summary

