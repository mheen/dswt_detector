import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from performance_tests.plot_dswt_check import plot_dswt_check
from ocean_model_data import select_input_files, load_roms_data
from transects import get_transects_in_lon_lat_range, get_specific_transect_data
from tools.files import get_dir_from_json
from tools import log
from datetime import datetime
import pandas as pd
import numpy as np
import xarray as xr

recheck_differences = True

year = 2017
model = 'cwa'
grid_file = f'{get_dir_from_json("cwa-roms")}grid.nc'

# --- Calculate total number of transects for model year
input_dir = f'{get_dir_from_json("cwa-roms")}{year}/'
files_contain = f'{model}_'
input_files = select_input_files(input_dir, file_contains=files_contain)

lon_range = [114.0, 116.0]
lat_range = [-33.0, -31.0]
transects_file = f'input/transects/{model}_transects.json'
transects = get_transects_in_lon_lat_range(transects_file, lon_range, lat_range)

n_files = len(input_files)
n_transects = len(transects)
n_times = len(xr.load_dataset(input_files[0]).ocean_time.values)
total_transects = n_files*n_transects*n_times

performance_file = f'performance_tests/{model}_{year}_performance_comparison.csv'

# --- Check performance
def check_performance(performance_file:str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(performance_file)

    manual_dswt = df['manual_dswt'].values
    algorithm_dswt = df['algorithm_dswt'].values

    l_comparison = manual_dswt == algorithm_dswt
    performance = np.sum(l_comparison)/len(manual_dswt)*100

    p_covered = len(df)/total_transects*100

    log.info(f'Algorithm performance: {np.round(performance, 1)}% accuracy based on {len(df)} tests, which is {np.round(p_covered, 2)}% of all available transects.')
    
    df_diff = df.loc[l_comparison == False]
    
    return df, df_diff

df, df_diff = check_performance(performance_file)

# --- Write differences to file
def write_differences_to_file(df_diff):
    diff_file = f'performance_tests/{model}_{year}_performance_differences.csv'
    
    df_diff.to_csv(diff_file, index=False)
    log.info(f'Wrote differences between manual and algorithm to csv file: {diff_file}')

# --- Check differences and change manual input if wanted
if recheck_differences == True:
    changes = 0
    for i in range(len(df_diff)):
        filename = df_diff["filename"].values[i]
        input_path = f'{input_dir}{filename}.nc'
        time_str = str(df_diff['time'].values[i])
        transect = df_diff['transect'].values[i]

        roms_ds = load_roms_data(input_path, grid_file=grid_file)
        roms_times = pd.to_datetime(roms_ds.ocean_time.values)
        time = datetime.strptime(time_str, '%Y%m%d%H%M')
        t = np.where(roms_times == time)[0][0]
        
        transect_ds = get_specific_transect_data(roms_ds, transects, transect)
        plot_dswt_check(transect_ds, t)
        
        manual_input_str = input('DSWT True/False (t/f): ')
        manual_input = True if manual_input_str.lower().startswith('t') else False
        if manual_input != df_diff['manual_dswt'].values[i]:
            l_row = np.logical_and(df['filename'] == filename, df['transect'] == transect)
            l_col = df.columns == 'manual_dswt'
            df.loc[l_row, l_col] = manual_input
            changes += 1
            
    # write performance comparison to file again if any changes
    if changes > 0:
        df.to_csv(performance_file, index=False)
        
        # redo performance check if any changes
        df, df_diff = check_performance(performance_file)
        write_differences_to_file(df_diff)
        
    else:
        log.info(f'Performance not changed after manual checks of differences')
        