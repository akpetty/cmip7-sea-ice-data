"""Ensemble statistics, anomalies, and monthly trend diagnostics."""
import warnings

import numpy as np
import xarray as xr
from scipy import signal, stats

from scripts.preprocessing.preprocessing import model_names

# ── Ensemble Statistics ────────────────────────────────────────────────────────

def _models_with_members(ds, thresh):
    """Return the source_ids in ds that have at least thresh ensemble members."""
    members_count = ds.member_id.groupby(model_names(ds)).count()
    return members_count.where(members_count >= thresh).dropna('member_id').member_id

def ensemble_mean(ds, thresh=1):
    """Compute per-model ensemble mean, requiring at least thresh members."""
    models = _models_with_members(ds, thresh)
    ens_mean = ds.groupby(model_names(ds)).mean()
    ens_mean = ens_mean.sel(member_id=models)
    ens_mean['member_id'] = ens_mean.member_id.astype(dtype='<U25')
    return ens_mean

def ensemble_std(ds, thresh=2):
    """Compute per-model ensemble std dev, requiring at least thresh members."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        models = _models_with_members(ds, thresh)
        ens_std = ds.groupby(model_names(ds)).std('member_id', ddof=1)
        ens_std = ens_std.sel(member_id=models)
        ens_std['member_id'] = ens_std.member_id.astype(dtype='<U25')
    return ens_std

def ensemble_count(ds, rename=True):
    """Count ensemble members per model."""
    members_count = ds.member_id.groupby(model_names(ds)).count()
    members_count['member_id'] = members_count.member_id.astype(dtype='<U25')
    if rename == True:
        members_count = members_count.rename({'member_id':'source_id'})
    return members_count.to_dataset(name='members')

def int_variability(ds, thresh=2, dims='member_id'):
    """Compute intra-ensemble variance per model as a proxy for internal variability."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = ds.dropna(dim='member_id', how='all')
        models = _models_with_members(ds, thresh)
        int_var = ds.groupby(model_names(ds)).std(dims, ddof=1)**2
        int_var = int_var.sel(member_id=models)
        int_var['member_id'] = int_var.member_id.astype(dtype='<U25')
    return int_var

# ── Time and Anomaly Operations ────────────────────────────────────────────────

def linear_detrend_xarray(da, dim='time'):
    """Apply linear detrending along the specified dimension."""
    return xr.apply_ufunc(
        signal.detrend,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        kwargs={'type': 'linear'},
        dask='parallelized')

def calc_anom(ds, dim='time', stand=False, detrend=True, ens_mean_cli=False):
    """Compute anomalies along a dimension, optionally after detrending and standardizing.

    dim='time' removes a monthly climatology; dim='year' removes the mean over years.
    ens_mean_cli=True uses the multi-model mean of per-model ensemble means as the climatology.
    """
    if dim not in ['time', 'year']:
        return None
    if detrend == True:
        ds = linear_detrend_xarray(ds, dim)
    if dim == 'time':
        def climatology(x):
            """Mean seasonal cycle of x."""
            return x.groupby('time.month').mean()
        grouped = ds.groupby('time.month')
    else:
        def climatology(x):
            """Mean of x over years."""
            return x.mean('year')
        grouped = ds
    if ens_mean_cli == True and 'member_id' in ds.dims:
        cli = climatology(ensemble_mean(ds)).mean('member_id')
    else:
        cli = climatology(ds)
    if stand == True:
        std = ds.groupby('time.month').std() if dim == 'time' else ds.std('year')
        return xr.apply_ufunc(lambda x, m, s: (x - m) / s, grouped, cli, std, dask='allowed')
    return grouped - cli

# ── Monthly Trend Decomposition ────────────────────────────────────────────────

def _apply_by_month(da, func_1d, time_dim='time'):
    """Apply a 1-D reduction over time_dim separately for each calendar month; returns a new 'month' dim (1-12)."""
    results = []
    for m in range(1, 13):
        grp = da.isel({time_dim: (da[time_dim].dt.month == m).values})
        out = xr.apply_ufunc(
            func_1d, grp,
            input_core_dims=[[time_dim]],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
        )
        results.append(out.expand_dims(month=[m]))
    return xr.concat(results, dim='month')

def monthly_linear_trend(da, time_dim='time', scale=1.0):
    """
    Per-calendar-month linear trend slope across years.

    Returns a DataArray with the same non-time dimensions plus a new 'month'
    coordinate (1-12). Multiply each slope by `scale` before returning
    (e.g. scale=10 converts units/year to units/decade).
    """
    def _slope_1d(x):
        """Least-squares slope of x against its index (NaN if fewer than 3 finite values)."""
        t = np.arange(len(x), dtype=float)
        mask = np.isfinite(x)
        if mask.sum() < 3:
            return np.nan
        return stats.linregress(t[mask], x[mask]).slope * scale

    return _apply_by_month(da, _slope_1d, time_dim)


def monthly_detrended_climo(da, time_dim='time'):
    """
    Remove a per-calendar-month linear trend, then compute the climatological mean.

    Returns a DataArray with the same non-time dimensions plus a new 'month'
    coordinate (1-12). Detrending is done independently for each calendar month,
    so the result is the mean seasonal cycle with the long-term trend removed.
    """
    def _detrend_mean_1d(x):
        """Mean of x after removing its linear trend (plain mean if fewer than 3 finite values)."""
        t = np.arange(len(x), dtype=float)
        mask = np.isfinite(x)
        if mask.sum() < 3:
            return np.nanmean(x)
        s, intercept, *_ = stats.linregress(t[mask], x[mask])
        return np.nanmean(x - (s * t + intercept))

    return _apply_by_month(da, _detrend_mean_1d, time_dim)
