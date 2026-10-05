# CMIP7 Sea Ice Data

A collaborative repository for wrangling and analyzing CMIP7 sea ice output in partnership with members of SIMIP.

This project is intended to support reproducible workflows for:
- discovering and accessing CMIP7 sea-ice model output
- preprocessing and quality-control on model data products
- exploratory analysis in notebooks
- diagnostic summaries and comparison workflows
- figure generation for reports, papers, and working-group outputs

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
│   ├── README.md
│   ├── 00_getting_started/
│   │   └── README.md
│   ├── 01_data_access/
│   │   └── README.md
│   ├── 02_processing/
│   │   └── README.md
│   ├── 03_analysis/
│   │   └── README.md
│   └── 04_plotting/
│       └── README.md
├── scripts/
│   ├── README.md
│   ├── data_access/
│   │   └── README.md
│   ├── preprocessing/
│   │   └── README.md
│   ├── diagnostics/
│   │   └── README.md
│   └── plotting/
│       └── README.md
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── README.md
│   ├── processed/
│   │   └── README.md
│   └── metadata/
│       └── README.md
├── results/
│   ├── README.md
│   ├── figures/
│   │   └── README.md
│   ├── outputs/
│   │   └── README.md
│   └── summaries/
│       └── README.md
├── docs/
│   ├── README.md
│   ├── conventions.md
│   ├── data_access.md
│   └── workflow.md
├── config/
│   ├── README.md
│   ├── s3_paths.yml
│   └── project_defaults.yml
├── src/
│   └── README.md
└── tests/
    └── README.md
```

## Data access and storage

This repository is designed around a clear separation between:
- source data access
- local processing and QC workflows
- derived outputs and summary products
- figure generation

Operationally, the project is intended to work with cloud-hosted CMIP7 outputs in an S3-compatible environment, while keeping the repository itself lightweight and portable. Bucket-specific paths and conventions should be tracked in `config/s3_paths.yml` instead of being hardcoded into analysis notebooks.

## Using UV

This project is configured for a lightweight Python workflow using `uv`.

### Install

```bash
uv sync
```

### Create a virtual environment

```bash
uv venv
source .venv/bin/activate
```

### Add a dependency

```bash
uv add xarray netcdf4 dask matplotlib
```

### Run a notebook or script

```bash
uv run jupyter lab
```

or

```bash
uv run python scripts/preprocessing/example_script.py
```

## Working conventions

- Keep analysis notebooks focused on exploration, diagnostics, and visualization.
- Put reusable logic in `scripts/` and package-level utilities in `src/` as the project grows.
- Store generated products in `results/` rather than committing derived files to version control.
- Document data provenance and processing steps in `docs/` or notebook markdown cells.
- Keep configuration paths (including S3 locations) in `config/`.

## Collaboration notes

This repository is intended to support a collaborative workflow with SIMIP contributors and other partners working on CMIP7 sea ice diagnostics. The structure is designed to be open to shared analysis, but still maintain clear conventions for data handling, plotting, and output generation.

## License

This project is provided under the MIT License unless otherwise noted.
