# Data access scripts

- `load_cmip.py`: the `CMIP6` and `CMIP7` classes, which search the Pangeo cloud and ESGF catalogs, download or open the files, and optionally regrid them or reduce them to sector means/sums. Derived variables (freeboard, ice density, ...) are computed from their inputs after loading.
- `ancillary.py`: loads the static files in `data/ancillary/`: target grids, NSIDC-0771 cell areas, and NSIDC-0780 sea ice regions.
