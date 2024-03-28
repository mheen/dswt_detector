import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.files import get_dir_from_json, create_dir_if_does_not_exist
from readers.read_ocean_data import load_roms_data, select_input_files
from transects import get_transects_dict_from_json, get_transects_in_lon_lat_range, get_specific_transect_data
from dswt.dswt_detection import determine_dswt_along_transect
from plot_tools.dswt import transects_plot
from tools import log

import matplotlib.pyplot as plt
from datetime import date
import random
import pandas as pd
import numpy as np
import os
import glob

# --------------------------------------------------------
# User input
# --------------------------------------------------------
n_files_to_check = 30
n_times_to_check = 2
n_transects_per_file_to_check = 5

year = 2017
model = 'cwa'
focus_months = [5, 6, 7] # set to None for full year,
# allowing this option to focus more on DSWT times
# rather than confirming obvious False values

input_dir = f'{get_dir_from_json("cwa-roms")}{year}/'
files_contain = f'{model}_'
grid_file = f'{get_dir_from_json("cwa-roms")}grid.nc'

transects_file = f'input/transects/{model}_transects.json'

lon_range = [114.0, 116.0]
lat_range = [-33.0, -31.0]

output_file = f'performance_tests/output/{model}_{year}_performance_comparison.csv'

# --------------------------------------------------------
# Randomly select file and transect and get user DSWT
# --------------------------------------------------------
transects = get_transects_in_lon_lat_range(transects_file, lon_range, lat_range)
transect_names = list(transects.keys())

if focus_months is not None:
    input_files = []
    for m in focus_months:
        m_str = str(m).zfill(2)
        input_files = input_files + glob.glob(f'{input_dir}*{files_contain}*{year}{m_str}*.nc')
else:
    input_files = select_input_files(input_dir, file_contains=files_contain)

for i in range(n_files_to_check):
    input_path = random.choice(input_files)
    filename = os.path.splitext(os.path.split(input_path)[1])[0]
    roms_ds = load_roms_data(input_path, grid_file=grid_file)
    
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
            transect_name = random.choice(transect_names)
            
            # don't select transect again if it is already in the output file
            if os.path.exists(output_file):
                df_old = pd.read_csv(output_file)
                df_file = df_old.loc[df_old['filename'] == filename]
                if not df_file.empty:
                    while transect_name in df_file['transect'].values:
                        transect_name = random.choice(transect_names)

            transect_ds = get_specific_transect_data(roms_ds, transects, transect_name)

            fig = plt.figure(figsize=(6, 8))
            fig = transects_plot(transect_ds, t, fig, 3, 2, 1, set_vlim=False)
            plt.show()
            
            manual_input = input('DSWT True/False (t/f): ')
            manual_dswt = True if manual_input.lower().startswith('t') else False
            l_dswt, condition1, condition2, drhodz_max, drhodz_cells = determine_dswt_along_transect(transect_ds)
            
            data = np.array([filename, time, transect_name, manual_dswt, l_dswt[t],
                             condition1[t], condition2[t], drhodz_max[t], drhodz_cells[t]])
            df = pd.DataFrame(np.expand_dims(data, 0),
                              columns=['filename', 'time', 'transect', 'manual_dswt', 'algorithm_dswt',
                                       'negative_drhodx', 'drhodz_condition', 'drhodz_max', 'drhodz_p_cells'])

            # append to csv file immediately after input
            if os.path.exists(output_file):
                df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df.to_csv(output_file, index=False)

log.info(f'Done comparing manual and algorithm DSWT, wrote/added results to file: {output_file}')
