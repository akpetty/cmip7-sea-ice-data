# CMIP7 Sea Ice Data

A collaborative repository for wrangling and analyzing CMIP7 sea ice output in partnership with members of SIMIP.

This project is intended to support reproducible workflows for:
- discovering and accessing CMIP7 sea-ice model output
- preprocessing and quality-control on model data products
- exploratory analysis in notebooks
- diagnostic summaries and comparison workflows
- figure generation for reports, papers, and working-group outputs

## History

This repository was created to help SIMIP contributors work with CMIP7 sea ice data in a consistent and reproducible way. The initial focus is on making remote data access straightforward and providing a lightweight set of shared analysis and plotting workflows that can be adapted for collaborative diagnostics.

The project is intentionally lightweight and exploratory at this stage: most work is centered on notebooks, reproducible data-access checks, and shared plotting patterns rather than a large package API.

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
│   ├── 01_data_access/
│   │   └── README.md
│   └── 02_plotting/
│       └── README.md
├── scripts/
├── data/
├── results/
├── docs/
├── config/
├── src/
└── tests/
```

## Data access and storage

This repository is designed around a clear separation between:
- source data access
- local processing and QC workflows
- derived outputs and summary products
- figure generation

Operationally, the project is intended to work with cloud-hosted CMIP7 outputs in an S3-compatible environment, while keeping the repository itself lightweight and portable. Bucket-specific paths and access notes should be documented in the notebook workflow and configuration files rather than hard-coded into analysis outputs.

## Using UV

The project environment and Python dependency list are managed in `pyproject.toml`. This is the canonical place where package information is stored and updated for the UV workflow.

For the normal developer workflow, use the project environment directly:

```bash
uv sync
source .venv/bin/activate
jupyter lab
```

This creates the virtual environment, activates it in your current shell, and lets you run Python, notebooks, and scripts as usual. If you prefer, you can also run commands without activating the environment first:

```bash
uv run jupyter lab
uv run python scripts/preprocessing/example_script.py
```

### Add a dependency

```bash
uv add xarray netcdf4 dask matplotlib
```

## Working conventions

- Keep analysis notebooks focused on exploration, diagnostics, and visualization.
- Put reusable logic in `scripts/` and package-level utilities in `src/` as the project grows.
- Store generated products in `results/` rather than committing derived files to version control.
- Document data provenance and processing steps in `docs/` or notebook markdown cells.
- Keep configuration paths (including S3 locations) in `config/`.

## Contributing

Contributions are welcome as this project grows. This section is intentionally lightweight for now, but the expectation is that contributors will help improve the shared CMIP7 workflows and keep the repository easy to use for others.

Suggested workflow:
- open an issue to propose a new notebook, data-access workflow, or plotting idea
- create a branch for your change
- keep notebook content focused and well documented
- prefer clear, reproducible code over one-off analysis steps
- submit a pull request with a short description of the change and any assumptions

If you are adding new analysis or plotting notebooks, follow the existing structure and keep the repository’s early workflow simple and readable.

## Collaboration notes

This repository is intended to support a collaborative workflow with SIMIP contributors and other partners working on CMIP7 sea ice diagnostics. The structure is designed to be open to shared analysis, common data-access routines, and reproducible figure generation.

## License

This project is provided under the MIT License unless otherwise noted.
