from tools.files import get_dir_from_json, create_dir_if_does_not_exist
from tools.config import read_config
from performance_tests.manual_random_checks import manual_performance_checks
from performance_tests.rate_performance import plot_performance_summary, plot_monthly_performance, recheck_differences
from transects import read_transects_in_lon_lat_range_from_json

import numpy as np
import xarray as xr
import pandas as pd
import os

# --------------------------------------------------------
# User input
# --------------------------------------------------------
model = 'cwa'
year = 2017 # year to check performance of DSWT detection

# --- Domain range
lon_range = [114.0, 116.0] # set to None for full domain
lat_range = [-33.0, -31.0] # set to None for full domain

# --- DSWT detection settings (default config file: 'input/configs/main_config.toml')
config = read_config(model)

# --- Input file info
input_dir = f'{get_dir_from_json("cwa")}{year}/'
grid_file = f'{get_dir_from_json("cwa")}grid.nc'
grid_ds = xr.open_dataset(grid_file) # loading grid: do not change!

transects_file = f'input/transects/{model}_transects.json'

# --- Performance info
n_files_to_check = 10
n_times_to_check = 2
n_transects_per_file_to_check = 5

focus_months = [5, 6, 7] # set to None for full year,
# allowing this option to focus more on DSWT times
# rather than confirming obvious false values

# option to be shown transects where manual and algorithm did not match again
recheck = False

# --- Output files (no need to change)
output_comparison = f'performance_tests/output/{model}/{model}_{year}_performance_comparison.csv'
output_performance = f'performance_tests/output/{model}/{model}_performance_summary.jpg'
output_performance_monthly = f'performance_tests/output/{model}/{model}_performance_monthly.jpg'
create_dir_if_does_not_exist(os.path.dirname(output_comparison))

# --------------------------------------------------------
# Manual checks
# --------------------------------------------------------
transects = read_transects_in_lon_lat_range_from_json(transects_file, lon_range, lat_range)

manual_performance_checks(input_dir, grid_file, config, year, transects,
                          focus_months, n_files_to_check, n_transects_per_file_to_check, n_times_to_check,
                          output_comparison)

df = pd.read_csv(output_comparison)

# --------------------------------------------------------
# Recheck differences
# --------------------------------------------------------
if recheck == True:
    recheck_differences(input_dir, grid_file, transects, df)

# --------------------------------------------------------
# Plot performance
# --------------------------------------------------------
plot_performance_summary(df, output_path=output_performance)
plot_monthly_performance(df, output_path=output_performance_monthly)
