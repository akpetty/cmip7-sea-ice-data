"""Regrid model output to a target grid with xESMF, during loading or after it."""
import warnings

import cf_xarray as cfxr
import numpy as np
import xarray as xr
import xesmf as xe

# Atmospheric variables: regridded bilinearly and never masked to the ocean
ATMOS_VARS = ['tas', 'ts']
# Models on unstructured grids: regridded with nearest_s2d from a locstream
UNSTRUCTURED_SIDS = ['ICON-ESM-LR', 'AWI-CM-1-1-MR', 'AWI-ESM-1-1-LR']


class RegridWarning(UserWarning):
    """Warning from regrid() that ds is missing its ocean mask or areacello."""


def add_corners(ds, verbose=False):
    """Build lon_b/lat_b cell corners on a curvilinear grid.

    Unwraps longitude to a monotonic branch along x before differencing, so a
    seam that sits at a different x index in every row (as on eORCA1) does not
    corrupt the bounds. Corners are wrapped back to [0, 360) at the end.
    """
    ds3 = ds.copy()

    xdim = 'x' if 'x' in ds.lon.dims else ds.lon.dims[-1]
    axis = ds.lon.dims.index(xdim)

    lon_raw = np.asarray(ds.lon.values, dtype=float)
    lon_uw = np.rad2deg(np.unwrap(np.deg2rad(lon_raw), axis=axis))

    if verbose:
        d = np.abs(np.diff(lon_uw, axis=axis))
        print(f"post-unwrap max |dlon| along {xdim}: {d.max():.3f} deg")
        print(f"unwrapped range: {lon_uw.min():.1f} to {lon_uw.max():.1f}")

    ds3 = ds3.assign_coords(lon=(ds.lon.dims, lon_uw))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds3 = ds3.cf.add_bounds(['lon', 'lat'])

    chunks = {d: -1 for d in ('bounds', 'y', 'x') if d in ds3.lat_bounds.dims}
    lat_corners = cfxr.bounds_to_vertices(
        ds3.lat_bounds.chunk(chunks), "bounds", order=None)
    lon_corners = cfxr.bounds_to_vertices(
        ds3.lon_bounds.chunk(chunks), "bounds", order=None)

    lon_corners = lon_corners % 360.0
    lat_corners = lat_corners.clip(-90.0, 90.0)

    ds3 = ds3.assign_coords(lon=(ds.lon.dims, lon_raw))
    ds3 = ds3.assign_coords(lon_b=lon_corners, lat_b=lat_corners)
    ds3 = ds3.rename({'y_vertices': 'y_b', 'x_vertices': 'x_b'})
    ds3 = ds3.drop_vars(['lat_bounds', 'lon_bounds'])
    return ds3


def _prepare_mask(ds, warn):
    """Return ds with 'mask' as a 2-D ocean fraction (0-1) on the lat/lon dims.

    xESMF uses it as the source grid mask (cells where it is 0 are left out of the regridding).
    After load_data() it is raw sftof (%), possibly with extra dims, so it is reduced and rescaled.
    """
    if 'mask' not in ds:
        if warn:
            warnings.warn("regrid: no 'mask' (ocean fraction) found, so land cells are not left out "
                          "of the regridding. Load with CMIP6/CMIP7, which attach it, or add ds['mask'].",
                          RegridWarning, stacklevel=3)
        return ds
    mask = ds['mask']
    extra = [d for d in mask.dims if d not in ds.lon.dims]
    if extra:
        mask = mask.isel({d: 0 for d in extra}, drop=True)
    if float(mask.max()) > 1:
        mask = mask/100  # sftof is in %
    return ds.assign_coords(mask=mask) if 'mask' in ds.coords else ds.assign(mask=mask)


def _is_unstructured(ds):
    """True if ds is on an unstructured grid.

    The grid's shape decides first: lat/lon sharing one cell dimension is unstructured, and 2-D
    lat/lon on two dims longer than 1 is structured, whatever the model (so an unstructured-grid
    model that was already regridded is treated as structured). Only when the shape is ambiguous
    (e.g. a size-1 y dimension) does the model name in member_id decide, via UNSTRUCTURED_SIDS;
    a mix of unstructured and other models then raises ValueError.
    """
    if ds.lat.ndim == 1 and ds.lat.dims == ds.lon.dims:
        return True
    if ds.lat.ndim == 2 and all(ds.sizes[d] > 1 for d in ds.lat.dims):
        return False
    if 'member_id' in ds.coords:
        sids = {str(m).split('_')[0] for m in np.atleast_1d(ds.member_id.values)}
        unstr = sids & set(UNSTRUCTURED_SIDS)
        if unstr and unstr != sids:
            raise ValueError(f'regrid: {sorted(unstr)} are on unstructured grids and {sorted(sids - unstr)} '
                             'are not; regrid one model at a time')
        return bool(unstr)
    return False


def regrid(ds, target_grid, method='conservative_normed', unstructured=None, atmos=None, warn=True):
    """Regrid ds (Dataset or DataArray with 2-D lat/lon on y/x) to target_grid with xESMF.

    Steps:
      1. Check for 'mask' (ocean fraction, or sftof in %) and 'areacello', warning if missing.
         They are attached by CMIP6/CMIP7.load_data(). The mask is the source grid mask: cells
         where it is 0 (land) are left out of the regridding.
      2. Roll grids whose first column is a duplicate 360-degree column.
      3. Conservative methods: build lon_b/lat_b corners (add_corners) when missing, and rebuild
         them after rolling the seam to the edge when the existing corners are broken.
      4. Regrid. atmos=True (default: every variable is tas/ts) uses bilinear with nearest_s2d
         extrapolation and no mask; unstructured=True uses nearest_s2d from a locstream; otherwise
         `method` (bilinear also extrapolates with nearest_s2d). unstructured=None detects it
         from the grid's shape, falling back to the model name in member_id (see _is_unstructured).
      5. If target_grid has a 'mask', the output is masked to it. xESMF copies target_grid's 2-D
         coordinates (mask, areacello) onto the output, and target_grid's lon_b/lat_b corners
         are attached; the source mask, areacello and corners are not carried over.
    warn=False silences the step-1 warnings.
    """
    name = None
    if isinstance(ds, xr.DataArray):
        name = ds.name or 'data'
        ds = ds.to_dataset(name=name)
    if atmos is None:
        atmos = len(ds.data_vars) > 0 and all(v in ATMOS_VARS for v in ds.data_vars)
    if unstructured is None:
        unstructured = _is_unstructured(ds)

    # 1. mask and areacello
    if not atmos:
        ds = _prepare_mask(ds, warn)
    if warn and 'areacello' not in ds:
        warnings.warn("regrid: no 'areacello' found. It isn't needed to regrid, but data loaded with "
                      "CMIP6/CMIP7 have one, so check ds came from there (output of an earlier regrid "
                      "only has one if that target grid did).", RegridWarning, stacklevel=2)
    ds = ds.drop_vars('areacello', errors='ignore')

    # 2. dateline wrap on structured grids
    if not unstructured:
        if ds.lon.isel(x=0).mean() == 360 and ds.lon.isel(x=1).mean() == 1.:
            ds = ds.roll(x=-1, roll_coords=True)

    # 3./4. method parameters (and corners for conservative methods)
    locstream_in = False
    skipna = True
    extrap_method = None
    if atmos:
        method = 'bilinear'
        extrap_method = 'nearest_s2d'
    elif unstructured:
        method = 'nearest_s2d'
        locstream_in = True
    elif method == 'bilinear':
        if 'mask' in ds:
            ds['mask'] = ds['mask'].where(ds['mask'] > 0)
            ds['mask'].loc[dict(y=ds['mask'].y[-1])] = ds['mask'].loc[dict(y=ds['mask'].y[-1])].fillna(0)
        extrap_method = 'nearest_s2d'  # was inverse_dist; nearest is safer
    elif method in ['conservative_normed', 'conservative']:
        if 'lon_bounds' in ds.coords or 'lon_b' not in ds.coords:
            ds = add_corners(ds.drop_vars(['lon_bounds', 'lat_bounds', 'vertex', 'bounds'], errors='ignore'))
        elif (np.abs(ds.lon_b - 180) < 0.5).sum() > ds.y.size:
            # corners broken along the longitude seam: roll the seam to the edge and rebuild them
            lon_diff = np.abs(ds.lon.diff('x'))
            seam_x = int(lon_diff.mean('y').argmax('x').values)
            ds = ds.roll(x=-(seam_x + 1), roll_coords=True)
            ds = add_corners(ds.drop_vars(['lon_b', 'lat_b'], errors='ignore'))

    if not atmos and 'mask' in ds and ds['mask'].shape != ds.lon.shape:
        ds['mask'] = ds['mask'].transpose('x', 'y')

    regridder = xe.Regridder(ds, target_grid, method, ignore_degenerate=True,
                             extrap_method=extrap_method, periodic=True, locstream_in=locstream_in)
    ds_new = regridder(ds, skipna=skipna)

    # 5. target grid mask, and the target's cell corners in place of the source's (which xESMF
    # carries over because they are on their own y_b/x_b dims), so the output can be regridded again
    if not atmos and 'mask' in ds_new:
        ds_new = ds_new.where(ds_new['mask']).set_coords('mask')
    ds_new = ds_new.drop_vars(['lon_b', 'lat_b', 'y_b', 'x_b'], errors='ignore')
    if 'lon_b' in target_grid.coords and 'lat_b' in target_grid.coords:
        ds_new = ds_new.assign_coords(lon_b=target_grid.lon_b, lat_b=target_grid.lat_b)
    return ds_new[name] if name is not None else ds_new
