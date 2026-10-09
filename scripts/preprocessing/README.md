# Preprocessing scripts

- `regrid.py`: regridding with xESMF via `regrid(ds, target_grid)`. It builds cell corners on curvilinear grids, uses nearest-neighbour regridding for unstructured-grid models, and regrids atmospheric variables bilinearly without ocean masking.
- `preprocessing.py`: time-coordinate helpers, model and member selection, and region masks built from the NSIDC-0780 sea ice regions.
