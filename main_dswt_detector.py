from transects import generate_transects_json_file, read_transects_in_lon_lat_range_from_json
from guis.transect_removal import interactive_transect_removal
from guis.transect_addition import interactive_transect_addition
from guis.check_dswt_config import interactive_transect_time_cycling_plot

from performance_tests.manual_random_checks import manual_performance_checks
from performance_tests.rate_performance import check_performance, recheck_differences

from readers.read_ocean_data import load_roms_data, select_input_files

from dswt.dswt_detection import determine_daily_dswt_along_multiple_transects
from tools.dswt_output import get_domain_str
from tools.config import Config, read_config, print_config
from tools import log
from tools.files import get_dir_from_json, create_dir_if_does_not_exist
import os
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
from warnings import warn

# --------------------------------------------------------
# User input
# --------------------------------------------------------
model = 'cwa'
years = np.array([2017]) # years to detect DSWT
performance_year = 2017 # year to check performance of DSWT detection

# --- Domain range
lon_range = [114.0, 116.0] # set to None for full domain
lat_range = [-33.0, -31.0] # set to None for full domain

# --- DSWT detection settings (default config file: 'input/configs/main_config.toml')
config = read_config(model)

# --- Input file info
model_input_dir = get_dir_from_json('test_data', json_file='input/example_dirs.json')
grid_file = f'{model_input_dir}grid.nc' # set to None if grid information in output files
file_preface = f'{model}_'

# --------------------------------------------------------
# Optional file settings (no need to change)
transects_dir = 'input/transects/'
create_dir_if_does_not_exist(transects_dir)

# --- Output file info
domain = get_domain_str(lon_range, lat_range)
output_dir = f'output/{model}_{domain}/'
create_dir_if_does_not_exist(output_dir)

# using transects for entire model domain and then selecting
# only relevant ones within requested domain range
# (using this method because cutting of the model domain
# to generate transects can go wrong when determining the land polygon)
transects_file = f'{transects_dir}{model}_transects.json'

# --------------------------------------------------------
# 1. Create and/or read in transects
# --------------------------------------------------------
log.info('''----------------------------------------------
               Creating transects
            ----------------------------------------------''')
# create transects and save to .json file if file does not already exist
if not os.path.exists(transects_file):
    grid_ds = xr.open_dataset(grid_file)
    generate_transects_json_file(grid_ds, config, transects_file)
    
    # plot to check transects and remove obviously faulty ones
    removed_transects = interactive_transect_removal(transects_file, grid_ds, config,
                                                     lon_range=lon_range,
                                                     lat_range=lat_range)
    
    # add transects in specific regions and from specific contour (useful when there are islands)
    added_transects = interactive_transect_addition(transects_file, grid_ds, config,
                                  lon_range=lon_range, lat_range=lat_range)
    if added_transects == True:
        # plot removal again to see if any of the added transects need to be removed
        interactive_transect_removal(transects_file, grid_ds, config,
                                     lon_range=lon_range,
                                     lat_range=lat_range)
    
else:
    log.info(f'Transects file already exists, using existing file: {transects_file}')

transects = read_transects_in_lon_lat_range_from_json(transects_file, lon_range, lat_range)

# --------------------------------------------------------
# 2. Input files check
# --------------------------------------------------------
log.info('''----------------------------------------------
               Checking input file variables and format
            ----------------------------------------------''')
# check that files contain required variables (for 1 file)
input_dir = f'{model_input_dir}{years[0]}/'
roms_files = select_input_files(input_dir, file_preface=file_preface)

# assuming curvilinear ROMS grid (does not work on other grids yet)
required_vars = ['ocean_time', 's_rho', 's_w',
                 'Vtransform', 'Cs_r', 'Cs_w', 'hc',
                 'angle', 'lon_rho', 'lat_rho', 'h',
                 'temp', 'salt', 'u', 'v']

ds_roms = load_roms_data(roms_files[0], grid_file=grid_file)
vars = list(ds_roms.keys()) + list(ds_roms.coords)
for v in required_vars:
    if not v in vars:
        raise ValueError(f'Missing required ROMS variable: {v}')

# check that files contain daily data (does not allow any other input file format yet)
if len(ds_roms.ocean_time) > 0:
    hours = (pd.to_datetime(ds_roms.ocean_time.values[-1])-pd.to_datetime(ds_roms.ocean_time.values[0])).total_seconds()/(60*60)
    if hours > 24.0:
        raise ValueError(f'ROMS input files contain data spanning more than 1 day. Please convert input files to daily data.')
    else:
        log.info('Passed file check.')
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

print_config(config)
bool_str = input('Would you like to check these settings? y/n: ')
if bool_str.lower().startswith('y'):
    interactive_transect_time_cycling_plot(model_input_dir, grid_file, transects, config)
else:
    log.info('Continuing with DSWT detection with current config.')

# --------------------------------------------------------
# 4. Performance check
# --------------------------------------------------------
performance_file = f'performance_tests/output/{model}_{performance_year}_performance_comparisonb.csv'
diff_file = f'performance_tests/output/{model}_{performance_year}_performance_differencesb.csv'

def _check():
    _, _ = check_performance(performance_file, diff_file)
    bool_str = input('Do you want to recheck differences? y/n: ')
    if bool_str.lower().startswith('y'):
        recheck_differences(f'{model_input_dir}{performance_year}/', grid_file, transects,
                            performance_file, diff_file)

if os.path.exists(performance_file):
    _check()      
else:
    bool_check = input('No manual performance comparison exists yet. Would you like to run this? y/n: ')
    if bool_check.lower().startswith('y'):
        n_files = int(input('How many files do you want to check?'))
        n_transects_per_file = int(input('How many transects per file?'))
        n_times_to_check = int(input('How many times per transect per file?'))
        manual_performance_checks(f'{model_input_dir}{performance_year}/', grid_file, model, performance_year,
                                    transects, None, n_files, n_transects_per_file, n_times_to_check,
                                    performance_file)
        _check()
    else:
        log.info('''You can run performance checks using "performance_tests/manual_random_checks"
                    and "performance_tests/rate_performance" if you want to do this later.''')

# --------------------------------------------------------
# 5. Detect DSWT & cross-shelf DSWT transport
# --------------------------------------------------------
log.info('''----------------------------------------------
               Detecting DSWT
            ----------------------------------------------''')
for year in years:
    log.info(f'Detecting DSWT for {year}')
    input_dir = f'{model_input_dir}{year}/'
    output_dswt = f'{output_dir}dswt_{year}.csv'
    
    if os.path.exists(output_dswt):
        df_temp = pd.read_csv(output_dswt)
        time = df_temp['time'].values
        time_last = datetime.strptime(pd.unique(time)[-1], '%Y-%m-%d')
        if time_last == datetime(year, 12, 31):
            log.info(f'Output already exists for {year}, skipping.')
            continue
        else:
            log.info(f'''Output partially exists for {year}. Running from {time_last+timedelta(days=1)} onwards.
                     Please check to make sure that all transects for {time_last} were written to file.''')
            date_range = [time_last+timedelta(days=1), datetime(year, 12, 31)]
    else:
        date_range = [datetime(year, 1, 1), datetime(year, 12, 31)]
    
    roms_files = select_input_files(input_dir, file_preface=file_preface, date_range=date_range)
    roms_files.sort()

    for file in roms_files:
        # Load ROMS data
        ds_roms = load_roms_data(file, grid_file, drop_vars=drop_vars)
        
        df_transects_dswt = determine_daily_dswt_along_multiple_transects(ds_roms, transects, config)
        
        if os.path.exists(output_dswt):
            df_transects_dswt.to_csv(output_dswt, mode='a', header=False, index=False)
        else:
            df_transects_dswt.to_csv(output_dswt, index=False)

# --------------------------------------------------------
# Output: timeseries and maps analyses and plots
# --------------------------------------------------------

# # to read csv and then get daily means:
# df = pd.read_csv(output_dswt, index_col=['time', 'transect']) # (this is a MultiIndex DataFrame)
# df.groupby('time').mean()

# # to select specific values:
# df.xs('2017-01-01', level='time', drop_level=False).xs('t153', level='transect', drop_level=False)