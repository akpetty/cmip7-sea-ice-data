# Ancillary data

Small static files used by the loading and preprocessing code. They are loaded by `scripts/data_access/ancillary.py`.

| Folder | File | Contents | Used for |
|---|---|---|---|
| `grids/` | `CESM2_grid.nc` | CESM2 ocean grid (lat/lon, corners, `areacello`, `mask`) | Target grid for unstructured-grid models (`grid_CESM2`) |
| `grids/` | `IS2_grid.nc` | ICESat-2 / NSIDC 25 km polar stereographic NH grid | `grid_ATL20_nh` (`new_grid=`) |
| `grids/` | `ATL20_grid.nc` | ICESat-2 ATL20 25 km polar stereographic SH grid | `grid_ATL20_sh` (`new_grid=`) |
| `grids/` | `OSISAF_nh_grid.nc`, `OSISAF_sh_grid.nc` | OSI SAF EASE2 25 km grids | `grid_OSISAF_nh`, `grid_OSISAF_sh` (`new_grid=`) |
| `cell_area/` | `NSIDC0771_CellArea_PS_{N,S}25km_v1.0.nc` | NSIDC-0771 polar stereographic 25 km cell areas | `ATL20_area_NH`, `ATL20_area_SH` |
| `region_masks/` | `NSIDC-0780_SeaIceRegions_{NH,SH-NASA}_v1.0.*` | NSIDC-0780 sea ice region shapefiles | `region_mask()`, sector means/sums (`NH_seaice_regions`, `SH_seaice_regions`) |
