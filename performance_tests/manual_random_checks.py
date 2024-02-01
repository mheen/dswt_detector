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
n_files_to_check = 20
n_transects_per_file_to_check = 3

year = 2017
model = 'cwa'

input_dir = f'{get_dir_from_json("cwa-roms")}{year}/'
files_contain = f'{model}_'
grid_file = f'{get_dir_from_json("cwa-roms")}grid.nc'

transects_file = f'input/transects/{model}_transects.json'

lon_range = [114.0, 116.0]
lat_range = [-33.0, -31.0]

output_file = f'performance_tests/{model}_{year}_performance_comparison.csv'

# --------------------------------------------------------
# Randomly select file and transect and get user DSWT
# --------------------------------------------------------
transects = get_transects_in_lon_lat_range(transects_file, lon_range, lat_range)
transect_names = list(transects.keys())

input_files = select_input_files(input_dir, file_contains=files_contain)

filenames = []
transect_headers = []
manual_dswt = []
algorithm_dswt = []
algorithm_condition1 = []
algorithm_condition2 = []
algorithm_drhodz_max = []
algorithm_drhodz_cells = []
for i in range(n_files_to_check):
    input_path = random.choice(input_files)
    filename = os.path.splitext(os.path.split(input_path)[1])[0]
    roms_ds = load_roms_data(input_path, grid_file=grid_file)
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
        plot_dswt_check(transect_ds, 0)
        
        filenames.append(filename)
        transect_headers.append(transect_name)
        
        manual_input = input('DSWT True/False (t/f): ')
        manual_dswt.append(True if manual_input.lower().startswith('t') else False)
        l_dswt, condition1, condition2, drhodz_max, drhodz_cells = determine_dswt_along_transect(transect_ds)
        algorithm_dswt.append(l_dswt[0])
        algorithm_condition1.append(condition1[0])
        algorithm_condition2.append(condition2[0])
        algorithm_drhodz_max.append(drhodz_max[0])
        algorithm_drhodz_cells.append(drhodz_cells[0])

filenames = np.array(filenames)
transect_headers = np.array(transect_headers)
manual_dswt = np.array(manual_dswt)
algorithm_dswt = np.array(algorithm_dswt)
algorithm_condition1 = np.array(algorithm_condition1)
algorithm_condition2 = np.array(algorithm_condition2)
algorithm_drhodz_max = np.array(algorithm_drhodz_max)
algorithm_drhodz_cells = np.array(algorithm_drhodz_cells)
        
# --------------------------------------------------------
# Append manual and algorithm DSWT result to file
# --------------------------------------------------------
# --- Write to file
df = pd.DataFrame(np.array([filenames, transect_headers, manual_dswt, algorithm_dswt,
                            algorithm_condition1, algorithm_condition2, algorithm_drhodz_max, algorithm_drhodz_cells]).transpose(),
                  columns=['filename', 'transect', 'manual_dswt', 'algorithm_dswt',
                           'negative_drhodx', 'drhodz_condition', 'drhodz_max', 'drhodz_p_cells'])

if os.path.exists(output_file):
    df.to_csv(output_file, mode='a', header=False, index=False)
else:
    df.to_csv(output_file, index=False)
log.info(f'Wrote performance to file: {output_file}')
