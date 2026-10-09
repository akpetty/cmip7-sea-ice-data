"""Static ancillary data (target grids, cell areas, sea ice region masks) in data/ancillary/."""
from pathlib import Path

import geopandas as gp
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
ANCILLARY_DIR = REPO_ROOT / 'data' / 'ancillary'


def ancillary_path(*parts):
    """Return the path to a file under data/ancillary/, e.g. ancillary_path('grids', 'CESM2_grid.nc')."""
    return ANCILLARY_DIR.joinpath(*parts)


# Target grids for regridding (pass as new_grid=)
grid_CESM2 = xr.open_dataset(ancillary_path('grids', 'CESM2_grid.nc'))
grid_ATL20_nh = xr.open_dataset(ancillary_path('grids', 'IS2_grid.nc'))
grid_ATL20_sh = xr.open_dataset(ancillary_path('grids', 'ATL20_grid.nc'))
grid_OSISAF_nh = xr.open_dataset(ancillary_path('grids', 'OSISAF_nh_grid.nc'))
grid_OSISAF_sh = xr.open_dataset(ancillary_path('grids', 'OSISAF_sh_grid.nc'))

# NSIDC-0771 polar stereographic 25 km cell areas
ATL20_area_NH = xr.open_dataset(ancillary_path('cell_area', 'NSIDC0771_CellArea_PS_N25km_v1.0.nc')).cell_area
ATL20_area_SH = xr.open_dataset(ancillary_path('cell_area', 'NSIDC0771_CellArea_PS_S25km_v1.0.nc')).cell_area

# NSIDC-0780 sea ice regions
NH_seaice_regions = gp.read_file(ancillary_path('region_masks', 'NSIDC-0780_SeaIceRegions_NH_v1.0.shp'))
SH_seaice_regions = gp.read_file(ancillary_path('region_masks', 'NSIDC-0780_SeaIceRegions_SH-NASA_v1.0.shp'))
