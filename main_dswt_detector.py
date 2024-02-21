from read_ocean_data import load_roms_data, select_roms_subset, select_input_files
from transects import generate_transects_json_file, get_transects_dict_from_json, get_transects_in_lon_lat_range
from tools.dswt_detection import calculate_mean_dswt_along_all_transects
from plot_tools.basic_timeseries import plot_histogram_monthly_dswt

from tools import log
from tools.files import get_dir_from_json, create_dir_if_does_not_exist
import os
import numpy as np
from datetime import datetime
import pandas as pd
import xarray as xr

# --------------------------------------------------------
# User input
# --------------------------------------------------------
# --- Input file info
main_input_dir = get_dir_from_json('cwa-roms')
years = np.arange(2000, 2024)
model = 'cwa'

grid_file = f'{main_input_dir}grid.nc'

files_contain = f'{model}_' # set to None if not needed

transects_dir = 'input/transects/'
create_dir_if_does_not_exist(transects_dir)

# --- Output file info
output_dir = 'output/'
create_dir_if_does_not_exist(output_dir)

# --- Domain range
lon_range = [114.0, 116.0] # set to None for full domain
lat_range = [-33.0, -31.0] # set to None for full domain

# --- DSWT detection settings
# no need to change if using recommended settings
if model.lower() == 'cwa':
    minimum_drhodz = 0.02
    minimum_p_cells = 0.10
    drhodz_depth_p = 0.50
    filter_depth = 100.
elif model.lower() == 'ozroms': # note: for daily ozroms
    minimum_drhodz = 0.01
    minimum_p_cells = 0.10
    drhodz_depth_p = 0.50
    filter_depth = 100.
else: # default settings
    minimum_drhodz = 0.02
    minimum_p_cells = 0.10
    drhodz_depth_p = 0.50
    filter_depth = 100.

# --- Automated file naming (no need to change)
lon_range_str = f'{int(np.floor(lon_range[0]))}-{int(np.ceil(lon_range[1]))}'
lon_range_unit = 'E' if lon_range[0] > 0 else 'W'
lat_range_str = f'{int(abs(np.floor(lat_range[0])))}-{int(abs(np.ceil(lat_range[1])))}'
lat_range_unit = 'S' if lat_range[0] < 0 else 'S'
domain = f'{lon_range_str}{lon_range_unit}_{lat_range_str}{lat_range_unit}'

# using transects for entire model domain and then selecting
# only relevant ones within requested domain range
# (using this method because cutting of the model domain
# to generate transects can go wrong when determining the land polygon)
transects_file = f'{transects_dir}{model}_transects.json'

# --------------------------------------------------------
# Create transects
# --------------------------------------------------------
# create transects and save to .json file if file does not already exist
if not os.path.exists(transects_file):
    grid_ds = xr.open_dataset(grid_file)
    generate_transects_json_file(grid_ds, transects_file)
else:
    log.info(f'Transects file already exists, using existing file: {transects_file}')

# --------------------------------------------------------
# Detect dense shelf water transport
# --------------------------------------------------------
def write_daily_mean_dswt_fraction_to_csv(input_dir:str, files_contain:str, grid_file:str,
                                          transects_file:str, output_file:str,
                                          lon_range=None, lat_range=None):
    
    if lon_range is not None and lat_range is not None:
        transects = get_transects_in_lon_lat_range(transects_file, lon_range, lat_range)
    else:
        transects = get_transects_dict_from_json(transects_file)
    
    roms_files = select_input_files(input_dir, files_contain)
    roms_files.sort()

    time = []
    f_dswt = []
    for file in roms_files:
        # Load ROMS data
        ds = load_roms_data(file, grid_file)
        
        if lon_range is not None and lat_range is not None: # does this make computation faster?
            ds = select_roms_subset(ds, time_range=None, lon_range=lon_range, lat_range=lat_range)
            
        # Get daily mean percentage of DSWT occurrence along transects
        # !!! FIX !!! assuming here that each file contains daily data -> keep? but include check somewhere?
        f_dswt.append(calculate_mean_dswt_along_all_transects(ds, transects))
        ocean_time0 = pd.to_datetime(ds.ocean_time.values[0])
        time.append(datetime(ocean_time0.year, ocean_time0.month, ocean_time0.day))

    # Write to output file
    log.info(f'Writing daily fraction DSWT occurrence to file: {output_file}')
    time = np.array(time).flatten()
    f_dswt = np.array(f_dswt).flatten()
    df = pd.DataFrame(np.array([time, f_dswt]).transpose(), columns=['time', 'f_dswt'])
    df.to_csv(output_file, index=False)

# --- Detect DSWT occurrence and write to csv if file does not already exist
for year in years:
    input_dir = f'{main_input_dir}{year}/'
    output_file = f'{output_dir}{model}_{year}_{domain}.csv'
    
    if not os.path.exists(output_file):
        write_daily_mean_dswt_fraction_to_csv(input_dir, files_contain, grid_file,
                                            transects_file, output_file,
                                            lon_range=lon_range, lat_range=lat_range)
    else:
        log.info(f'Output file already exists, using existing file: {output_file}')
