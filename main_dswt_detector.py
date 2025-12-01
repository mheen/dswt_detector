from transects import generate_transects_json_file, read_transects_in_lon_lat_range_from_json
from guis.transect_removal import interactive_transect_removal
from guis.transect_addition import interactive_transect_addition, write_added_transect_keys_to_file

from readers.read_ocean_data import load_roms_data, select_input_files
from readers.read_dswt_output import get_domain_str, read_dswt_occurrence_timeseries, read_dswt_transport, calculate_transport_across_contour

from main_plots import plot_dswt_timeseries, plot_dswt_map

from dswt.dswt_detection import determine_daily_dswt_along_multiple_transects
from dswt.spatial_interpolation import spatially_interpolate_dswt
from tools.config import Config, read_config
from tools import log
from tools.files import get_dir_from_json, create_dir_if_does_not_exist

from scipy.interpolate import LinearNDInterpolator
import os
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
from warnings import warn

# --------------------------------------------------------
# User input
# --------------------------------------------------------
model = 'test'
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
grid_ds = xr.open_dataset(grid_file) # loading grid: do not change!
file_preface = f'{model}_' # set to None if files don't have a string preface

# --- Processing info
# Determine if transport should be interpolated to cover full grid range
spatially_interpolate_results = True # set to False if not required
# choose interpolator function to use (make sure this is a 2D interpolator and that is imported)
interpolator = LinearNDInterpolator

# --- Plot info
# Two plots will be created at the end of this script
# One shows a timeseries of monthly mean DSWT occurrence and transport across a depth contour
# The other shows a map of the overall mean cross-shelf transport
# Specify the depth contour for the timeseries and the output folder to save plots to here
depth_contour = 50.0
plot_dir = get_dir_from_json('plots', json_file='input/example_dirs.json')
create_dir_if_does_not_exist(plot_dir)

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
transects_file = f'{transects_dir}{model}_transects.json'

# --------------------------------------------------------
# 1. Create and/or read in transects
# --------------------------------------------------------
log.info('''----------------------------------------------
               Creating transects
            ----------------------------------------------''')
# create transects and save to .json file if file does not already exist
if not os.path.exists(transects_file):
    generate_transects_json_file(grid_ds, config, transects_file)
    
    # plot to check transects and remove obviously faulty ones
    removed_transects = interactive_transect_removal(transects_file, grid_ds, config,
                                                     lon_range=lon_range,
                                                     lat_range=lat_range)
    
    # add transects in specific regions and from specific contour (useful when there are islands)
    added_transects_bool, added_from_index = interactive_transect_addition(transects_file, grid_ds, config,
                                  lon_range=lon_range, lat_range=lat_range)
    if added_transects_bool == True:
        islands_file = f'{transects_dir}{model}_transects_islands.csv'
        write_added_transect_keys_to_file(transects_file, added_from_index, islands_file)
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
# redundant code: using relative values, but potentially
# print config parameters used now?
# depth filter and depth percentage

# maybe also print how performance checks can be done:
# I think this should be a fully separate thing now though

# --------------------------------------------------------
# 4. Detect DSWT & cross-shelf DSWT transport
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
# 5. Processing (spatial interpolation if desired)
# --------------------------------------------------------
log.info('''--------------------------------------------------
                         Processing output
            --------------------------------------------------''')
output_dir_processing = f'{output_dir}processed/'
create_dir_if_does_not_exist(output_dir_processing)

for year in years:
    
    output_transport = f'{output_dir_processing}dswt_transport_{year}.csv'
    output_timeseries = f'{output_dir_processing}dswt_timeseries_{year}.csv'
    
    # --- Process transport per time per location
    # Note: might be worth making these nc files instead??
    if os.path.exists(output_transport):
        log.info(f'Processed transport file exists, skipping: {output_transport}')
    else:
        df_transport = read_dswt_transport(output_dir, years, grid_ds)
        
        if spatially_interpolate_results == True:
            warn(f'Spatial interpolation set to True, this may take a while... set to False if not needed.')
            
            df_interp = spatially_interpolate_dswt(df_transport, interpolator, grid_ds, config)
            df_transport = df_interp
            
        # write transport per time per location to csv
        log.info(f'Writing processed DSWT transport to csv: {output_transport}')
        df_transport.to_csv(output_transport, index=False)
    
    # --- Process timeseries
    if os.path.exists(output_timeseries):
        log.info(f'Processed timeseries file exists, skipping: {output_timeseries}')
    else:
        df_transport = pd.read_csv(output_transport)
        # occurrence timeseries
        time, f_dswt = read_dswt_occurrence_timeseries(output_dir, years)
        # transport timeseries
        time, transport_contour, contour_length = calculate_transport_across_contour(df_transport,
                                                                                     grid_ds,
                                                                                     lon_range,
                                                                                     lat_range,
                                                                                     depth_contour,
                                                                                     dx_method='roms')

        df_timeseries = pd.DataFrame(data=np.array([time, f_dswt, transport_contour]).transpose(),
                                     columns=['time', 'f_dswt', f'transport_{str(int(depth_contour))}m'])
        log.info(f'Writing processed DSWT timeseries to csv: {output_timeseries}')
        df_timeseries.to_csv(output_timeseries, index=False)   

# --------------------------------------------------------
# Output: timeseries and maps analyses and plots
# --------------------------------------------------------
log.info('''--------------------------------------------------
                              Creating plots
            --------------------------------------------------''')

df_transport = pd.DataFrame(columns=['time', 'eta', 'xi', 'transport', 'mean_thickness', 'max_distance'])
df_timeseries = pd.DataFrame(columns=['time', 'f_dswt', f'transport_{str(int(depth_contour))}m'])
for year in years:
    output_transport = f'{output_dir_processing}dswt_transport_{year}.csv'
    df_transport_y = pd.read_csv(output_transport)
    df_transport = pd.concat([df_transport, df_transport_y])

    output_timeseries = f'{output_dir_processing}dswt_timeseries_{year}.csv'
    df_timeseries_y = pd.read_csv(output_timeseries)
    df_timeseries = pd.concat([df_timeseries, df_timeseries_y])

timeseries_plot = f'{plot_dir}{model}_timeseries.jpg'
plot_dswt_timeseries(df_timeseries, depth_contour, years,
                     output_path=timeseries_plot, show=False)
log.info(f'Saved timeseries plot to {timeseries_plot}')

map_plot = f'{plot_dir}{model}_map.jpg'
plot_dswt_map(df_transport, grid_ds, output_path=map_plot, show=False)
log.info(f'Saved map plot to {map_plot}')
