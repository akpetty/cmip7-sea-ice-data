"""Data availability summaries: which models/members are published for each variable and experiment."""
from functools import reduce

import numpy as np
import pandas as pd

from scripts.preprocessing.preprocessing import to_pystr_list

# ── CMIP6 Catalog and Model Discovery ─────────────────────────────────────────

def preferred_load_list(cat_cloud, cat_esgf):
    """Split models into cloud-preferred and ESGF lists based on catalog availability."""
    cloud_models = reduce(np.union1d, [cat_cloud[t].df.source_id.unique() for t in ['tgt', 'awgt', 'aswgt']])
    esgf_models = reduce(np.union1d, [cat_esgf[t].df.source_id.unique() for t in ['tgt', 'awgt', 'aswgt']])
    all_models = np.union1d(cloud_models, esgf_models)
    cloud_subset = np.setdiff1d(all_models, esgf_models)
    return {
        'cloud': to_pystr_list(cloud_subset),
        'esgf': to_pystr_list(esgf_models),
    }

# ── CMIP7 availability ────────────────────────────────────────────────────────

def cmip7_member_counts(summary, native_only=True):
    """Ensemble members per model, per experiment_id, from CMIP7.data_summary() (all ESGF).

    Returns {experiment_id: {source_id: number of distinct variant_labels}}. A member
    published on several grids or brandings counts once.
    """
    df = pd.concat(summary.values(), ignore_index=True)
    if native_only:
        df = df[df['native_grid'].eq(True) & ~df['region_grid'].eq(True)]
    n = df.groupby(['experiment_id', 'source_id'])['variant_label'].nunique()
    return {exp: {str(sid): int(k) for sid, k in g.droplevel(0).sort_index().items()}
            for exp, g in n.groupby(level=0)}

def cmip7_load_list(summary, native_only=True):
    """Models with data for one variable, per experiment_id, from CMIP7.data_summary() (all ESGF).

    Search with experiment_id=None to cover every published experiment. Returns
    {experiment_id: [source_id, ...]}, one entry (column) per experiment found.
    """
    return {exp: list(m) for exp, m in cmip7_member_counts(summary, native_only).items()}

# Column order for availability tables: historical, scenarios (low -> high forcing), then
# control/idealized runs; experiments not listed go last, alphabetically.
_CMIP7_EXP_ORDER = ['historical', 'esm-hist',
                    'scen7-vl', 'scen7-ln', 'scen7-l', 'scen7-ml', 'scen7-m', 'scen7-hl', 'scen7-h',
                    'esm-scen7-vl', 'esm-scen7-ln', 'esm-scen7-l', 'esm-scen7-ml', 'esm-scen7-m',
                    'esm-scen7-hl', 'esm-scen7-h',
                    'piControl', 'esm-piControl', '1pctCO2', 'abrupt-4xCO2']

def _sort_experiments(exps):
    """Sort experiment_ids in _CMIP7_EXP_ORDER; unlisted experiments go last, alphabetically."""
    rank = {e: i for i, e in enumerate(_CMIP7_EXP_ORDER)}
    return sorted(exps, key=lambda e: (rank.get(e, len(rank)), e))

def cmip7_availability_table(summaries, cmip6_names=None, fallback=None, native_only=True,
                             counts=False, members=True, sep=', '):
    """Variable x experiment_id table of available models from CMIP7.data_summary() results.

    summaries: {variable: data_summary() dict}, searched with experiment_id=None.
    Rows are labelled (variable_id, region) as published on ESGF, so a CMIP6-style name
    like sivoln shows as ('sivol', 'nh'); gridded fields are region 'glb'. Rows keep the
    order of summaries. long_name/units come from ESGF (the STAC index, via data_summary). Variables no model
    has published are left out, unless they are in fallback {variable_id: 'Long Name [units]'}:
    then the row is kept with that description and empty experiment columns.
    cmip6_names {cmip7 variable_id: cmip6 variable_id} adds a cmip6_name column.
    Cells are model names joined by sep ('' if none), each followed by its ensemble member
    count, e.g. 'UKESM1-3-LL (3)' (members=False drops it), or model counts if counts=True.
    """
    fallback = fallback or {}
    rows = {}
    for var, summary in summaries.items():
        by_exp = cmip7_member_counts(summary, native_only=native_only)
        if not by_exp and var not in fallback:
            continue
        df = pd.concat(summary.values(), ignore_index=True)
        info = df[['long_name', 'units']].dropna().drop_duplicates()
        if len(info):
            long_name, units = info.iloc[0]
        else:
            # 'Long Name [units]' -> ('Long Name', 'units')
            long_name, _, units = fallback.get(var, '').rpartition(' [')
            long_name, units = (long_name, units.rstrip(']')) if long_name else (units, None)
        # CMIP7 variable_id/region actually searched (e.g. sivoln -> sivol, nh); fallback rows keep var
        row = {'variable_id': ', '.join(df['variable_id'].dropna().unique()) or var,
               'region': ', '.join(sorted(df['region'].dropna().unique()))}
        if cmip6_names is not None:
            row['cmip6_name'] = cmip6_names.get(var)
        row.update({'long_name': long_name or None, 'units': units})
        for exp, models in by_exp.items():
            if counts:
                row[exp] = len(models)
            else:
                row[exp] = sep.join(f'{sid} ({k})' if members else sid for sid, k in models.items())
        rows[var] = row
    table = pd.DataFrame.from_dict(rows, orient='index').reindex(list(rows))  # keep input variable order
    meta = (['cmip6_name'] if cmip6_names is not None else []) + ['long_name', 'units']
    exps = _sort_experiments([c for c in table.columns if c not in meta + ['variable_id', 'region']])
    table = table.reindex(columns=['variable_id', 'region'] + meta + exps)
    table = table.fillna({e: 0 if counts else '' for e in exps}).set_index(['variable_id', 'region'])
    if counts:
        table[exps] = table[exps].astype(int)
    return table
