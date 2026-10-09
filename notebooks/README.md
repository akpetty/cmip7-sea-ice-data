# CMIP7 Sea Ice Data notebooks

Notebooks for exploring, processing, and analyzing CMIP7 sea ice output. Reusable code lives in `scripts/`; each notebook adds the repo root to `sys.path` so it can import from there.

| Folder | Notebook | What it does |
|---|---|---|
| `01_data_access/` | `cmip7_data_availability_and_loading.ipynb` | Checks which CMIP7 sea ice variables, models, and experiments are published on ESGF, and loads one variable as an example |
| `02_processing/` | `cmip7_regridding.ipynb` | Regrids CMIP7 output to a target grid, either while loading or afterwards |
| `03_analysis/` | `cmip7_cmip6_arctic_sic.ipynb` | Compares CMIP7 Arctic sea ice with CMIP6 and OSI SAF observations |

The first code cell of each notebook pip-installs the extra packages it needs (xESMF, intake-esgf, regionmask, ...). Downloaded model files go to `/tmp` and can be removed in the clean-up cell at the end.

Keep notebooks readable and annotate any assumptions, provenance, or data source details.
