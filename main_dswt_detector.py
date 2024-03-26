from readers.read_ocean_data import load_roms_data, select_roms_subset, select_input_files
from transects import generate_transects_json_file, get_transects_dict_from_json, get_transects_in_lon_lat_range
from dswt.dswt_detection import calculate_mean_dswt_along_all_transects
from tools.config import Config, read_config
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
model = 'cwa'
# years = np.arange(2000, 2024)
years = [2017]

# --- Domain range
lon_range = [114.0, 116.0] # set to None for full domain
lat_range = [-33.0, -31.0] # set to None for full domain

# --- DSWT detection settings (default config file: 'input/configs/main_config.toml')
config = read_config(model)

# --- Input file info
model_input_dir = get_dir_from_json('cwa')
grid_file = f'{model_input_dir}grid.nc' # set to None if grid information in output files
files_contain = f'{model}_' # set to None if not needed

# --------------------------------------------------------
# Optional file settings (no need to change)
transects_dir = 'input/transects/'
create_dir_if_does_not_exist(transects_dir)

# --- Output file info
output_dir = f'output/{model}/'
create_dir_if_does_not_exist(output_dir)

# --- Automated domain name
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
# Detect dense shelf water transport &
# Calculate total cross-shelf transport
# --------------------------------------------------------
# --- Detect DSWT occurrence and write to csv if file does not already exist
for year in years:
    input_dir = f'{model_input_dir}{year}/'
    output_dswt = f'{output_dir}dswt_{model}_{year}_{domain}.csv'
    
    if not os.path.exists(output_dswt):
        transects = get_transects_in_lon_lat_range(transects_file, lon_range, lat_range)
        
        roms_files = select_input_files(input_dir, files_contain)
        roms_files.sort()
        
        time = []
        f_dswt = []
        cross_dswt = []
        vel_dswt = []
        cross_bottom = []
        cross_surface = []
        cross_interior = []
        for file in roms_files:
            # Load ROMS data
            ds_roms = load_roms_data(file, grid_file)
            
            if lon_range is not None and lat_range is not None: # does this make computation faster?
                ds_roms = select_roms_subset(ds_roms, time_range=None, lon_range=lon_range, lat_range=lat_range)
                
            # Get daily mean percentage of DSWT occurrence along transects
            # !!! FIX !!! assuming here that each file contains daily data -> keep? but include check somewhere?
            ocean_time0 = pd.to_datetime(ds_roms.ocean_time.values[0])
            f_dswt_daily, transport_daily, vel_daily = calculate_mean_dswt_along_all_transects(ds_roms, transects, config)
            
            data = np.array([datetime(ocean_time0.year, ocean_time0.month, ocean_time0.day),
                             f_dswt_daily, transport_daily, vel_daily])
            columns = ['time', 'f_dswt', 'dswt_transport', 'dswt_velocity']
            df = pd.DataFrame(np.expand_dims(data, 0), columns=columns)
            
            if os.path.exists(output_dswt):
                df.to_csv(output_dswt, mode='a', header=False, index=False)
            else:
                df.to_csv(output_dswt, index=False)

    else:
        log.info(f'Output file already exists, using existing file: {output_dswt}')
