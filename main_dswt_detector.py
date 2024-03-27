from readers.read_ocean_data import load_roms_data, select_roms_subset, select_input_files
from transects import generate_transects_json_file, get_transects_dict_from_json, get_transects_in_lon_lat_range
from dswt.dswt_detection import determine_daily_dswt_along_multiple_transects
from dswt.cross_shelf_transport import calculate_daily_mean_cross_shelf_transport_at_depth_contour
from tools.config import Config, read_config
from tools import log
from tools.files import get_dir_from_json, create_dir_if_does_not_exist
import os
import numpy as np
from datetime import datetime
import pandas as pd
import xarray as xr
from warnings import warn

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

# --- Automated domain name
lon_range_str = f'{int(np.floor(lon_range[0]))}-{int(np.ceil(lon_range[1]))}'
lon_range_unit = 'E' if lon_range[0] > 0 else 'W'
lat_range_str = f'{int(abs(np.floor(lat_range[0])))}-{int(abs(np.ceil(lat_range[1])))}'
lat_range_unit = 'S' if lat_range[0] < 0 else 'S'
domain = f'{lon_range_str}{lon_range_unit}_{lat_range_str}{lat_range_unit}'

# --- Output file info
output_dir = f'output/{model}_{domain}/'
create_dir_if_does_not_exist(output_dir)

# using transects for entire model domain and then selecting
# only relevant ones within requested domain range
# (using this method because cutting of the model domain
# to generate transects can go wrong when determining the land polygon)
transects_file = f'{transects_dir}{model}_transects.json'

# --------------------------------------------------------
# 1. Create transects
# --------------------------------------------------------
# create transects and save to .json file if file does not already exist
if not os.path.exists(transects_file):
    grid_ds = xr.open_dataset(grid_file)
    generate_transects_json_file(grid_ds, transects_file)
else:
    log.info(f'Transects file already exists, using existing file: {transects_file}')
    
# add GUI option to show transects and remove faulty ones?
# report sanity check for width calculation

# --------------------------------------------------------
# 2. Input files check
# --------------------------------------------------------
# check that files contain required variables (for 1 file)
input_dir = f'{model_input_dir}{years[0]}/'
roms_files = select_input_files(input_dir, files_contain)

required_vars = ['ocean_time', 's_rho', 's_w',
                 'Vtransform', 'Cs_r', 'Cs_w', 'hc',
                 'angle', 'lon_rho', 'lat_rho', 'h',
                 'temp', 'salt', 'u', 'v']

ds_roms = load_roms_data(roms_files[0], grid_file=grid_file)
vars = list(ds_roms.keys()) + list(ds_roms.coords)
for v in required_vars:
    if not v in vars:
        raise ValueError(f'Missing required ROMS variable: {v}')

# check that files contain daily data
if len(ds_roms.ocean_time) > 0:
    hours = (pd.to_datetime(ds_roms.ocean_time.values[-1])-pd.to_datetime(ds_roms.ocean_time.values[0])).total_seconds()/(60*60)
    if hours > 24.0:
        raise ValueError(f'ROMS input files contain data spanning more than 1 day. Please convert input files to daily data.')
else:
    warn('Cannot determine if ROMS input files contain daily data. Please ensure they do.')

# get list of variables that can be dropped from reading
drop_vars = []
for v in vars:
    if v not in required_vars:
        drop_vars.append(v)

roms_files = None
ds_roms = None

# --------------------------------------------------------
# 3. Determine config parameters
# --------------------------------------------------------

# use from config file if existing for model
# otherwise run manual checks to determine values

# --------------------------------------------------------
# 4. Performance check
# --------------------------------------------------------

# report on DSWT detection performance
# run manual performance checks if wanted

# --------------------------------------------------------
# 5. Detect DSWT & cross-shelf DSWT transport
# --------------------------------------------------------
for year in years:
    input_dir = f'{model_input_dir}{year}/'
    output_dswt = f'{output_dir}dswt_{year}.csv'
    
    if os.path.exists(output_dswt):
        log.info(f'Output already exists for {year}, skipping.')
        continue
    
    transects = get_transects_in_lon_lat_range(transects_file, lon_range, lat_range)
    roms_files = select_input_files(input_dir, files_contain)
    roms_files.sort()

    for file in roms_files:
        # Load ROMS data
        ds_roms = load_roms_data(file, grid_file, drop_vars=drop_vars)
        
        if lon_range is not None and lat_range is not None: # does this make computation faster?
            ds_roms = select_roms_subset(ds_roms, time_range=None, lon_range=lon_range, lat_range=lat_range)
        
        df_transects_dswt = determine_daily_dswt_along_multiple_transects(ds_roms, transects, config)
        
        if os.path.exists(output_dswt):
            df_transects_dswt.to_csv(output_dswt, mode='a', header=False, index=False)
        else:
            df_transects_dswt.to_csv(output_dswt, index=False)

# # to read csv and then get daily means:
# df = pd.read_csv(output_dswt, index_col=['time', 'transect']) # (this is a MultiIndex DataFrame)
# df.groupby('time').mean()