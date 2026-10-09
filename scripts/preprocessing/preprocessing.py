"""Preprocessing helpers: time coordinates, model/member selection, and region masking."""
import regionmask
import xarray as xr

from scripts.data_access.ancillary import NH_seaice_regions, SH_seaice_regions

# NSIDC-0780 NH region indices that make up the Inner Arctic
INNER_ARCTIC_REGIONS = [0, 1, 2, 3, 4, 5, 6]

# ── Time coordinates ──────────────────────────────────────────────────────────

def preprocess_snap_to_month_start(ds):
    """Snap time coordinates to the first of each month for consistent merging."""
    new_time = ds.indexes['time'].to_period('M').to_timestamp()
    ds = ds.assign_coords(time=new_time)
    return ds

def to_monthly(ds):
    """Reshape a time-indexed dataset into (year, month) coordinates."""
    year = ds.time.dt.year
    month = ds.time.dt.month
    # assign new coords
    ds = ds.assign_coords(year=("time", year.data), month=("time", month.data))
    # reshape the array to (..., "month", "year")
    return ds.set_index(time=("year", "month")).unstack("time")

# ── Model selection utilities ─────────────────────────────────────────────────

def to_pystr_list(arr):
    """Convert a numpy array to a plain Python list of strings."""
    return arr.astype(str).tolist()

def model_names(ds):
    """Return the source_id part of each member_id ('CESM2_r1i1p1f1' -> 'CESM2'), for grouping by model."""
    return ds.member_id.str.split('split', '_').sel(split=0)

def sel_model(ds, sid='CESM2'):
    """Select ensemble members for a single model by source_id prefix."""
    subset = ds.sel(member_id=model_names(ds) == sid)
    return subset

def drop_sel(ds: xr.Dataset, sids=('CESM2-LE',)):
    """Drop models whose member_id prefix matches any of the given sids."""
    if ds is None:
        return None
    if isinstance(sids, (str, bytes)):
        sids = [sids]
    prefix = ds['member_id'].astype(str).str.replace(r'_.+', '', regex=True)
    keep = ~prefix.isin(sids)
    if 'member_id' in ds.dims:
        return ds.isel(member_id=keep)
    else:
        members_to_keep = ds['member_id'].where(keep, drop=True)
        return ds.sel(member_id=members_to_keep)

def first_member(ds, keep_member_id=False):
    """Return the first ensemble member per model."""
    first = ds.groupby(model_names(ds)).first()
    first['member_id'] = first.member_id.astype(dtype='<U25')
    return first

# ––––– Spatial Masking –––––––––––––––––

def region_mask(ds, region='Inner_Arctic'):
    """Mask a dataset to a named sea ice region.

    region is 'NH' (lat > 0), 'SH' (lat < 0), 'Arctic' (lat > 60), 'Antarctic' (lat < -60),
    'Inner_Arctic' (NSIDC-0780 NH regions 0-6; also 'Inner Arctic' or 'IA'), or a dict
    {'Arctic' or 'Antarctic': [NSIDC-0780 region indices]}. Any other value returns ds unmasked.
    """
    ds = ds.copy()
    if 'lat' in ds.coords and 'y' in ds.dims:
        if region in ['Inner Arctic', 'IA']:
            region = 'Inner_Arctic'
        if region == 'NH':
            ds_subset = ds.where(ds.lat > 0)
        elif region == 'SH':
            ds_subset = ds.where(ds.lat < 0)
        elif region == 'Arctic':
            ds_subset = ds.where(ds.lat > 60)
        elif region == 'Antarctic':
            ds_subset = ds.where(ds.lat < -60)
        elif region == 'Inner_Arctic':
            df = NH_seaice_regions
            mask = regionmask.mask_geopandas(df, ds.lon, ds.lat, overlap=False)
            ds_subset = ds.where(mask.isin(INNER_ARCTIC_REGIONS))
        elif isinstance(region, dict):
            region_key = list(region.keys())[0]
            region_values = list(region.values())[0]
            if region_key == 'Arctic':
                df = NH_seaice_regions
            elif region_key == 'Antarctic':
                df = SH_seaice_regions
            else:
                raise ValueError(f"Unrecognized region dict key: {region_key}")
            mask = regionmask.mask_geopandas(df, ds.lon, ds.lat, overlap=False)
            ds_subset = ds.where(mask.isin(region_values))
        else:
            # If region doesn't match anything, just return unmasked
            ds_subset = ds
    else:
        print('No lat or y dim found: returning dataset')
        ds_subset = ds

    return ds_subset
