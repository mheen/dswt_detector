from ocean_model_data import load_roms_data, select_roms_subset
from generate_transects import generate_transects_json_file
from dswt_detection import determine_dswt_along_multiple_transects
from gui_tools import plot_dswt_maps_transects

from tools import log
import os
import numpy as np

# --------------------------------------------------------
# User input
# --------------------------------------------------------
input_dir = 'tests/data/'
model = 'cwa'
grid_file = 'tests/data/cwa_grid.nc' # set to None if not needed
date = '20170623'
files_contain = f'{model}_{date}' # set to None if not needed

lon_range = [114.0, 116.0] # set to None for full domain
lat_range = [-33.0, -31.0] # set to None for full domain

lon_range_str = f'{int(np.floor(lon_range[0]))}-{int(np.ceil(lon_range[1]))}'
lon_range_unit = 'E' if lon_range[0] > 0 else 'W'
lat_range_str = f'{int(abs(np.floor(lat_range[0])))}-{int(abs(np.ceil(lat_range[1])))}'
lat_range_unit = 'S' if lat_range[0] < 0 else 'S'
domain_name = f'{model}_{lon_range_str}{lon_range_unit}_{lat_range_str}{lat_range_unit}'

transects_file = f'input/transects_{domain_name}.json'

# --------------------------------------------------------
# Detect dense shelf water transport
# --------------------------------------------------------

# --- Load ROMS data
ds = load_roms_data(input_dir, grid_file=grid_file, files_contain=files_contain)
if lon_range is not None and lat_range is not None:
    ds = select_roms_subset(ds, time_range=None, lon_range=lon_range, lat_range=lat_range)
    
# --- Create transects (saved to .json file)
if not os.path.exists(transects_file):
    generate_transects_json_file(ds, transects_file)
else:
    log.info(f'Transects file already exists, using existing file: {transects_file}')

# --- Find DSWT along transects
l_dswt = determine_dswt_along_multiple_transects(ds, transects_file)

# --- Check DSWT using interactive map and transect plotter
# Use keyboard arrows to cycle through time
# Use mouse to click on transects to plot transect data
plot_dswt_maps_transects(ds, transects_file, l_dswt, transect_interval=5,
                         lon_range=lon_range, lat_range=lat_range)
