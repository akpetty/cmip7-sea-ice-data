# Processing

Notebooks for preprocessing, regridding, and QC.

- `cmip7_regridding.ipynb`: regrids CanESM6-0-MR sea ice concentration to the OSI SAF 25 km NH grid, either by passing `new_grid=` to `CMIP7(...)` or by calling `regrid()` after loading, and checks how well sea ice area is conserved.
