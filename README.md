# CMIP7 Sea Ice Data



A collaborative repository for wrangling and analyzing CMIP7 sea ice output in partnership with members of the Sea Ice Model Intercomparison Project (SIMIP).

See the [Current contributors](#current-contributors) section below.

This project is intended to support reproducible workflows for:
- discovering and accessing CMIP7 sea-ice model output
- preprocessing and quality-control on model outputs (ensuring consistency and vetting of outputs).
- diagnostic summaries of data availability and applied corrections/fixes.
- exploratory data analysis with Jupyter Notebooks
- potential figure generation 

## History

This repository was created to facilitate efforts around data wrangling and processing to support CMIP7 sea ice data analysis. The initial focus is on making data access straightforward and providing [...]

The project is intentionally lightweight and exploratory at this stage: most work is centered on notebooks, reproducible data-access checks, and shared plotting options.

V0.1: October 5, 2026.
 - Initial template

V0.2: October 9, 2026.
 - CMIP6/CMIP7 loading, regridding, and diagnostics code in `scripts/`
 - Grids, cell areas, and region masks in `data/ancillary/`
 - First notebooks: CMIP7 data availability and loading, regridding, and a CMIP7 vs CMIP6 Arctic sea ice comparison


## Repository structure

```text
cmip7-sea-ice-data/
├── README.md
├── .gitignore
├── .python-version
├── pyproject.toml
├── environment.yml
├── requirements.txt
├── notebooks/
│   ├── 01_data_access/      # CMIP7 data availability and loading
│   ├── 02_processing/       # regridding
│   └── 03_analysis/         # CMIP7 vs CMIP6 Arctic sea ice
├── scripts/
│   ├── data_access/         # CMIP6/CMIP7 loaders, ancillary data
│   ├── preprocessing/       # regridding, time handling, region masks
│   ├── diagnostics/         # availability summaries, ensemble statistics
│   └── plotting/
├── data/
│   └── ancillary/           # target grids, cell areas, region masks
├── results/
│   ├── figures/
│   └── summaries/
├── docs/
├── config/
├── src/
└── tests/
```

## Current workflow

The code in `scripts/` covers the main steps from finding data to analysis:

1. **Find and load data**: the `CMIP6` and `CMIP7` classes in `scripts/data_access/load_cmip.py` search the Pangeo cloud and ESGF catalogs, then download or open the matching files as xarray datasets. 
2. **Regrid and process**: output can be regridded with xESMF (`scripts/preprocessing/regrid.py`) to a common grid, such as OSI SAF, ICESat-2, or CESM2, while loading or afterwards. Regional means and sums can also be computed.
3. **Summarize and analyze**: `scripts/diagnostics/` builds data availability tables and computes ensemble statistics, anomalies, and trends.

The notebooks in `notebooks/` walk through each step; see `notebooks/README.md`. Outputs they produce are saved under `results/`.

## Data access and storage

This repository is designed around a clear separation between:
- source/raw CMIP data access
- local processing, QC and generating derived datasets.
- uploads and access of derived data through AWS S3 buckets.

Operationally, the project is intended to work with both the raw (ESGF) and derived (AWS S3) datasets locally or on the cloud. The access of the S3 derived datasets should be much quicker/efficient if[...]

## Using UV

The project environment and Python dependency list are managed in `pyproject.toml`. This is where package information is stored and updated for the UV workflow.

For the normal developer workflow, use the project environment directly:

```bash
uv sync
source .venv/bin/activate
python scripts/preprocessing/example_script.py
```

This creates the virtual environment, activates it in your current shell, and lets you run Python, notebooks, and scripts as usual. If you prefer, you can also run commands without activating the envi[...]

```bash
uv run jupyter lab
uv run python scripts/preprocessing/example_script.py
```

### Add a dependency

```bash
uv add matplotlib
```

## Working conventions

- Keep analysis notebooks focused on exploration, diagnostics, and visualization.
- Put reusable logic in `scripts/` and package-level utilities in `src/` as the project grows.
- Store generated products in `results/` rather than committing derived files to version control.
- Document data provenance and processing steps in `docs/` or notebook markdown cells.
- Keep configuration paths (including S3 locations) in `config/`.

## Contributing

Contributions are welcome as this project grows. This repo is intentionally lightweight for now, but the expectation is that contributors will help improve the CMIP7 workflows and keep the repository [...]

Suggested workflow:
- open an issue to propose a new notebook, data-access workflow, or plotting idea
- create a branch for your proposed change
- keep notebook content focused and well documented
- prefer clear, reproducible code over one-off analysis steps
- submit a pull request with a short description of the change and any assumptions

If you are adding new analysis or plotting notebooks, follow the existing structure and keep the repository’s early workflow simple and readable.

See also the [Current contributors](#current-contributors) list below.

## Collaboration notes

This repository is intended to support a collaborative workflow with SIMIP contributors and other partners working on CMIP7 sea ice diagnostics. The structure is designed to be open to shared analysis[...]

## Current contributors

- Alek Petty ([@akpetty](https://github.com/akpetty))
- Chris Cardinale ([@cjcardinale](https://github.com/cjcardinale))

## License

This project is provided under the MIT License unless otherwise noted.
