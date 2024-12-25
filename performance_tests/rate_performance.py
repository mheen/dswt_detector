import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from performance_tests.plot_transects_for_manual_check import transects_plot
from readers.read_ocean_data import select_input_files, load_roms_data, select_roms_transect_from_known_coordinates
from transects import read_transects_in_lon_lat_range_from_json
from tools.files import get_dir_from_json
from tools import log
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import xarray as xr

def check_performance(performance_file:str, diff_file:str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(performance_file)

    manual_dswt = df['manual_dswt'].values
    algorithm_dswt = df['algorithm_dswt'].values

    l_comparison = manual_dswt == algorithm_dswt
    performance = np.sum(l_comparison)/len(manual_dswt)*100

    log.info(f'Algorithm performance: {np.round(performance, 1)}% accuracy based on {len(df)} tests.')
    
    df_diff = df.loc[l_comparison == False]
    
    false_positives = np.sum(df_diff['manual_dswt'] == False)
    false_negatives = np.sum(df_diff['manual_dswt'] == True)
    
    log.info(f'False positives: {false_positives/len(df_diff)*100}% - False negatives: {false_negatives/len(df_diff)*100}% of mistakenly detected DSWT.')
    
    df_diff.to_csv(diff_file, index=False)
    log.info(f'Wrote differences between manual and algorithm to csv file: {diff_file}')
    
    return df, df_diff

def recheck_differences(input_dir:str, grid_file:str, transects:dict,
                        performance_file:str, diff_file:str):
    
    df = pd.read_csv(performance_file)
    df_diff = pd.read_csv(diff_file)
    
    changes = 0
    for i in range(len(df_diff)):
        filename = df_diff["filename"].values[i]
        input_path = f'{input_dir}{filename}.nc'
        time_str = str(df_diff['time'].values[i])
        transect_name = df_diff['transect'].values[i]

        roms_ds = load_roms_data(input_path, grid_file=grid_file)
        roms_times = pd.to_datetime(roms_ds.ocean_time.values)
        time = datetime.strptime(time_str, '%Y%m%d%H%M')
        t = np.where(roms_times == time)[0][0]
        
        eta = transects[transect_name]['eta']
        xi = transects[transect_name]['xi']
        transect_ds = select_roms_transect_from_known_coordinates(roms_ds, eta, xi)
        transects_plot(transect_ds, t)
        plt.show()
        
        manual_input_str = input('DSWT True/False (t/f): ')
        manual_input = True if manual_input_str.lower().startswith('t') else False
        if manual_input != df_diff['manual_dswt'].values[i]:
            l_row = np.logical_and(df['filename'] == filename, df['transect'] == transect_name)
            l_col = df.columns == 'manual_dswt'
            df.loc[l_row, l_col] = manual_input
            changes += 1
            
    # write performance comparison to file again if any changes
    if changes > 0:
        df.to_csv(performance_file, index=False)
        
        # redo performance check if any changes
        df, df_diff = check_performance(performance_file)
        
        df_diff.to_csv(diff_file, index=False)
        log.info(f'Wrote differences between manual and algorithm to csv file: {diff_file}')
        
    else:
        log.info(f'Performance not changed after manual checks of differences')
        
if __name__ == '__main__':
    recheck = False

    year = 2017
    model = 'cwa'
    grid_file = f'{get_dir_from_json("cwa")}grid.nc'
    input_dir = f'{get_dir_from_json("cwa")}{year}/'

    lon_range = [114.0, 116.0]
    lat_range = [-33.0, -31.0]
    transects_file = f'input/transects/{model}_transects.json'
    transects = read_transects_in_lon_lat_range_from_json(transects_file, lon_range, lat_range)

    performance_file = f'performance_tests/output/{model}_{year}_performance_comparison.csv'
    diff_file = f'performance_tests/output/{model}_{year}_performance_differences.csv'

    df, df_diff = check_performance(performance_file)
    
    if recheck == True:
        recheck_differences(input_dir, grid_file, transects,
                            performance_file, diff_file)