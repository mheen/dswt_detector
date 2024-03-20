import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from readers.read_ocean_data import load_roms_data
from readers.read_meteo_data import load_era5_data, select_era5_in_closest_point, get_daily_mean_wind_data
from dswt.dswt_detection import determine_dswt_along_transect, determine_dswt_along_multiple_transects, calculate_horizontal_density_gradient_along_transect
from transects import get_specific_transect_data, get_transects_in_lon_lat_range
from gui_tools import plot_dswt_maps_transects, select_transects_to_plot
from performance_tests.plot_dswt_check import plot_dswt_scenario
from tools.files import get_dir_from_json, create_dir_if_does_not_exist
from datetime import datetime
import numpy as np

year = 2017

transects_file = 'input/transects/cwa_transects.json'
transects = get_transects_in_lon_lat_range(transects_file, [114.0, 116.0], [-33.0, -31.0])

main_input_dir = get_dir_from_json('cwa-roms')
input_dir = f'{main_input_dir}{year}/'
grid_file = f'{main_input_dir}grid.nc'

output_dir = 'performance_tests/plots/'
create_dir_if_does_not_exist(output_dir)

plot_lon_range = [114.5, 115.8]
plot_lat_range = [-33.0, -31.5]
meridians = [114.5, 115.5]
parallels = [-33.0, -32.0, -31.0]

dates = [datetime(2017, 1, 15), # no DSWT: positive drhodx
         datetime(2017, 2, 22), # no DSWT: negative drhodx because of higher salinity along coast, but vertically mixed water column
         datetime(2017, 6, 11), # DSWT
         datetime(2017, 9, 8), # manual DSWT, algorithm no DSWT (because not enough consecutive cells)
         datetime(2017, 5, 21)] # manual no DSWT, algorithm DSWT
transects_to_plot = ['t201', 't197', 't171', 't209', 't188']
time_to_plot = [0, 0, 6, 0, 5]

for i, date in enumerate(dates):
    output_path = f'{output_dir}cwa_{date.strftime("%Y%m%d")}_{transects_to_plot[i]}.jpg'
    
    # -- Load ROMS data
    input_path = f'{input_dir}cwa_{date.strftime("%Y%m%d")}_03__his.nc'
    roms_ds = load_roms_data(input_path, grid_file=grid_file)
    # # For interactive plotting:
    # transects_dswt = determine_dswt_along_multiple_transects(roms_ds, transects_file)
    # plot_dswt_maps_transects(roms_ds, transects_file, transects_dswt)
    
    transect_ds = get_specific_transect_data(roms_ds, transects, transects_to_plot[i])
    l_dswt, _, _, _, _ = determine_dswt_along_transect(transect_ds)
    
    # --- Load ERA5 wind data
    era5_ds = load_era5_data(get_dir_from_json("era5"), '2017')
    lon_p = transect_ds.lon_rho.values[np.floor(len(transect_ds.lon_rho)/2).astype(int)]
    lat_p = transect_ds.lat_rho.values[np.floor(len(transect_ds.lat_rho)/2).astype(int)]
    point_ds = select_era5_in_closest_point(era5_ds, lon_p, lat_p)
    time, _, _, wind_vel, wind_dir = get_daily_mean_wind_data(point_ds)
    i_wind = np.where(time == date)[0][0]
    
    plot_dswt_scenario(roms_ds, transect_ds, l_dswt, time_to_plot[i], wind_vel[i_wind], wind_dir[i_wind],
                       plot_lon_range, plot_lat_range, parallels, meridians,
                       output_path=output_path, show=False)