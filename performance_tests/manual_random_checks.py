import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.files import get_dir_from_json, create_dir_if_does_not_exist
from tools.config import Config
from readers.read_ocean_data import load_roms_data, select_input_files, select_roms_transect_from_known_coordinates
from transects import read_transects_in_lon_lat_range_from_json
from dswt.dswt_detection import determine_dswt_along_transect
from performance_tests.plot_transects_for_manual_check import transects_plot
from tools import log

import matplotlib.pyplot as plt
from datetime import date
import random
import pandas as pd
import numpy as np
import os
import glob

def manual_performance_checks(input_dir:str, grid_file:str, config:Config, year:int,
                              transects:dict, focus_months:list[int],
                              n_files_to_check:int, n_transects_per_file_to_check:int, n_times_to_check:int,
                              output_file:str):
    # --------------------------------------------------------
    # Randomly select file and transect and get user DSWT
    # --------------------------------------------------------
    transect_names = list(transects.keys())

    if focus_months is not None:
        input_files = []
        for m in focus_months:
            m_str = str(m).zfill(2)
            input_files = input_files + glob.glob(f'{input_dir}*{year}{m_str}*.nc')
    else:
        input_files = select_input_files(input_dir)

    selected_input_files = random.sample(input_files, n_files_to_check)
    selected_transect_names = random.sample(transect_names, n_transects_per_file_to_check)

    for i in range(n_files_to_check):
        filename = os.path.splitext(os.path.split(selected_input_files[i])[1])[0]
        roms_ds = load_roms_data(selected_input_files[i], grid_file=grid_file)
        
        if len(roms_ds.ocean_time) == 1:
            n_times_to_check = 1
        
        t_array = np.arange(0, len(roms_ds.ocean_time))
        t_previous = None
        t = None
        for k in range(n_times_to_check):
            while t == t_previous: # don't select same time multiple times
                t = random.choice(t_array)
                
            t_previous = t

            time = pd.to_datetime(roms_ds.ocean_time.values[t]).strftime('%Y%m%d%H%M')
            
            for j in range(n_transects_per_file_to_check):
                transect_name = selected_transect_names[j]
                
                # don't select transect again if it is already in the output file
                if os.path.exists(output_file):
                    df_old = pd.read_csv(output_file)
                    df_file = df_old.loc[df_old['filename'] == filename]
                    if not df_file.empty:
                        while transect_name in df_file['transect'].values:
                            transect_name = random.choice(transect_names)

                eta = transects[transect_name]['eta']
                xi = transects[transect_name]['xi']
                transect_ds = select_roms_transect_from_known_coordinates(roms_ds, eta, xi)
                transects_plot(transect_ds, t)
                plt.show()
                
                manual_dswt = input('DSWT 0=False, 1=True, 0.5=Unsure: ')
                transect_ds_t = transect_ds.isel(ocean_time=[t])
                (t_dswt, f_dswt, vel, thickness, transport,
                 distance, lon, lat, h, drhodx_mean, drhodx_min,
                 drhos_mean, drhos_min) = determine_dswt_along_transect(transect_ds_t, config)
                
                data = np.array([filename, time, transect_name, manual_dswt, f_dswt[0],
                                 np.nanmean(vel), np.nanmean(thickness), np.nanmean(transport),
                                 drhodx_mean, np.nanmean(drhos_mean)])
                df = pd.DataFrame(np.expand_dims(data, 0),
                                columns=['filename', 'time', 'transect', 'manual_dswt', 'algorithm_dswt',
                                         'vel', 'thickness', 'transport', 'drhodx', 'drho_s'])

                # append to csv file immediately after input
                if os.path.exists(output_file):
                    df.to_csv(output_file, mode='a', header=False, index=False)
                else:
                    df.to_csv(output_file, index=False)

    log.info(f'Done comparing manual and algorithm DSWT, wrote/added results to file: {output_file}')

if __name__ == '__main__':
    # --------------------------------------------------------
    # User input
    # --------------------------------------------------------
    n_files_to_check = 10
    n_times_to_check = 2
    n_transects_per_file_to_check = 5

    year = 2017
    model = 'cwa'
    focus_months = None # set to None for full year,
    # allowing this option to focus more on DSWT times
    # rather than confirming obvious false values

    input_dir = f'{get_dir_from_json("cwa")}{year}/'
    grid_file = f'{get_dir_from_json("cwa")}grid.nc'

    transects_file = f'input/transects/{model}_transects.json'

    lon_range = [114.0, 116.0]
    lat_range = [-33.0, -31.0]

    output_file = f'performance_tests/output/{model}_{year}_performance_comparison.csv'
    create_dir_if_does_not_exist(os.path.dirname(output_file))
    
    transects = read_transects_in_lon_lat_range_from_json(transects_file, lon_range, lat_range)
    
    manual_performance_checks(input_dir, grid_file, model, year, transects,
                              focus_months, n_files_to_check, n_transects_per_file_to_check, n_times_to_check,
                              output_file)