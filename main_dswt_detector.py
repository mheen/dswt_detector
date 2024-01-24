from ocean_model_data import load_roms_data, select_roms_subset, select_input_files
from generate_transects import generate_transects_json_file
from tools.dswt_detection import determine_dswt_along_multiple_transects
from process_dswt_detection import write_daily_mean_dswt_fraction_to_csv, calculate_monthly_mean_dswt_fraction
from gui_tools import plot_dswt_maps_transects

from tools import log
from tools.files import get_dir_from_json
import os
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

import matplotlib.pyplot as plt

# --------------------------------------------------------
# User input
# --------------------------------------------------------
# --- Input file info
main_input_dir = get_dir_from_json('cwa-roms')
year = '2017'
model = 'cwa'

input_dir = f'{main_input_dir}{year}/'
grid_file = 'tests/data/cwa_grid.nc'

files_contain = f'{model}_' # set to None if not needed

transects_dir = 'transects/'

# --- Output file info
output_dir = 'output/'

# --- Domain range
lon_range = [114.0, 116.0] # set to None for full domain
lat_range = [-33.0, -31.0] # set to None for full domain

# --- DSWT detection settings
# no need to change if using recommended settings
if model.lower() == 'cwa':
    minimum_drhodz = 0.02
    minimum_p_cells = 0.30
    filter_depth = 100.
elif model.lower() == 'ozroms': # note: for daily ozroms
    minimum_drhodz = 0.01
    minimum_p_cells = 0.30
    filter_depth = 100.
else: # default settings
    minimum_drhodz = 0.02
    minimum_p_cells = 0.30
    filter_depth = 100.

# --- Automated file naming (no need to change)
lon_range_str = f'{int(np.floor(lon_range[0]))}-{int(np.ceil(lon_range[1]))}'
lon_range_unit = 'E' if lon_range[0] > 0 else 'W'
lat_range_str = f'{int(abs(np.floor(lat_range[0])))}-{int(abs(np.ceil(lat_range[1])))}'
lat_range_unit = 'S' if lat_range[0] < 0 else 'S'
domain = f'{lon_range_str}{lon_range_unit}_{lat_range_str}{lat_range_unit}'

transects_file = f'{transects_dir}{model}_{domain}.json'

output_file = f'{output_dir}{model}_{year}_{domain}.csv'

# --------------------------------------------------------
# Create transects
# --------------------------------------------------------
# create transects and save to .json file if file does not already exist
if not os.path.exists(transects_file):
    generate_transects_json_file(grid_file, transects_file)
else:
    log.info(f'Transects file already exists, using existing file: {transects_file}')

# --------------------------------------------------------
# Detect dense shelf water transport
# --------------------------------------------------------
# detect DSWT along transects and write to csv if file does not already exist
if not os.path.exists(output_file):
    write_daily_mean_dswt_fraction_to_csv(input_dir, files_contain, grid_file,
                                          transects_file, output_file,
                                          lon_range=lon_range, lat_range=lat_range)
else:
    log.info(f'Output file already exists, using existing file: {output_file}')

# --------------------------------------------------------
# 
# --------------------------------------------------------
time, f_dswt = calculate_monthly_mean_dswt_fraction(output_file)
ax = plt.axes()
ax.bar(time, f_dswt)
plt.show()

# --------------------------------------------------------
# Interactive plots to check DSWT detection
# --------------------------------------------------------
# # --- Check DSWT using interactive map and transect plotter
# # Use keyboard arrows to cycle through time
# # Use mouse to click on transects to plot transect data
# plot_dswt_maps_transects(ds, transects_file, l_dswt, transect_interval=5,
#                          lon_range=lon_range, lat_range=lat_range)
