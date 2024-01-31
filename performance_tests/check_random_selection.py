import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.files import get_dir_from_json, create_dir_if_does_not_exist
from ocean_model_data import load_roms_data, select_input_files
from transects import get_transects_dict_from_json, get_transects_in_lon_lat_range, get_specific_transect_data
from tools.dswt_detection import determine_dswt_along_transect
from performance_tests.plot_dswt_check import plot_dswt_check
from tools import log

from datetime import date
import random
import pandas as pd
import numpy as np
import os

# --------------------------------------------------------
# User input
# --------------------------------------------------------
n_files_to_check = 2
n_transects_per_file_to_check = 1

year = 2017
model = 'cwa'

input_dir = f'{get_dir_from_json("cwa-roms")}{year}/'
files_contain = f'{model}_'
grid_file = f'{get_dir_from_json("cwa-roms")}grid.nc'

transects_file = f'input/transects/{model}_transects.json'

lon_range = [114.0, 116.0]
lat_range = [-33.0, -31.0]

output_dir = 'performance_tests/output/'
output_file = f'{output_dir}dswt_comparison_{model}_{year}_{date.today().strftime("%d-%m-%Y")}.csv'

# --------------------------------------------------------
# Randomly select file and transect and get user DSWT
# --------------------------------------------------------
transects = get_transects_in_lon_lat_range(transects_file, lon_range, lat_range)
transect_names = list(transects.keys())

input_files = select_input_files(input_dir, file_contains=files_contain)

description = []
manual_dswt = []
algorithm_dswt = []
for i in range(n_files_to_check):
    input_path = random.choice(input_files)
    roms_ds = load_roms_data(input_path, grid_file=grid_file)
    for j in range(n_transects_per_file_to_check):
        transect_name = random.choice(transect_names)
        transect_ds = get_specific_transect_data(roms_ds, transects, transect_name)
        
        plot_dswt_check(roms_ds, transect_ds, lon_range, lat_range, None, None)
        
        description.append(f'{os.path.splitext(os.path.split(input_path)[1])[0]}_{transect_name}')
        
        manual_input = input('DSWT True/False: ')
        manual_dswt.append(True if manual_input == 'True' else False)
        l_dswt = determine_dswt_along_transect(transect_ds)
        algorithm_dswt.append(l_dswt[0])

description = np.array(description)
manual_dswt = np.array(manual_dswt)
algorithm_dswt = np.array(algorithm_dswt)
        
# --------------------------------------------------------
# Compare manual and algorithm DSWT
# --------------------------------------------------------
# --- Write to file
df = pd.DataFrame(np.array([description, manual_dswt, algorithm_dswt]).transpose(), columns=['file_transect', 'manual_dswt', 'algorithm_dswt'])
create_dir_if_does_not_exist(output_dir)
df.to_csv(output_file, index=False)
log.info(f'Wrote performance to file: {output_file}')

# --- Print overall performance
overall_performance = np.round(np.sum(algorithm_dswt == manual_dswt)/len(description)*100, 1)

print(f'''Algorithm DSWT detection agreed with manual detection {overall_performance}% of the time
      based on {n_files_to_check} randomly selected files with {n_transects_per_file_to_check} randomly
      selected transects per file (so for {n_files_to_check*n_transects_per_file_to_check} random tests).''')
