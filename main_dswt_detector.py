from ocean_model_data import load_roms_data, select_roms_subset
from generate_transects import generate_transects_json_file
from dswt_detection import determine_dswt_along_multiple_transects, plot_cycling_map_with_transects

from tools import log
import os

# --------------------------------------------------------
# User input
# --------------------------------------------------------
input_dir = 'tests/data/'
model = 'ozroms'
grid_file = None # set to None if not needed
date = '20170623'
files_contain = f'{model}_{date}' # set to None if not needed

lon_range = [114.0, 116.0] # set to None for full domain
lat_range = [-33.0, -31.0] # set to None for full domain

domain_name = f'{model}_{lon_range[0]}-{lon_range[1]}E_{abs(lat_range[0])}-{abs(lat_range[1])}S'

transects_file = f'input/transects_{domain_name}.json'

# --------------------------------------------------------
# Detect dense shelf water transport
# --------------------------------------------------------

# Load ROMS data
ds = load_roms_data(input_dir, grid_file=grid_file, files_contain=files_contain)
if lon_range is not None and lat_range is not None:
    ds = select_roms_subset(ds, time_range=None, lon_range=lon_range, lat_range=lat_range)
    
# Create transects (saved to .json file)
if not os.path.exists(transects_file):
    generate_transects_json_file(ds, transects_file)
else:
    log.info(f'Transects file already exists, using existing file: {transects_file}')

# Find DSWT along transects
l_dswt = determine_dswt_along_multiple_transects(ds, transects_file)

plot_cycling_map_with_transects(ds, transects_file=transects_file, l_dswt=l_dswt,
                                lon_range=lon_range, lat_range=lat_range, transect_interval=5)