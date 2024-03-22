import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from dswt.dswt_detection import determine_dswt_along_transect
from dswt.cross_shelf_transport import calculate_dswt_cross_shelf_transport_along_transect, get_down_transect_velocity_component
from readers.read_ocean_data import load_roms_data, select_roms_subset, select_roms_transect
from transects import get_transects_in_lon_lat_range
from tools.files import get_dir_from_json
from tools.config import read_config
from plot_tools.dswt import plot_transect, plot_vertical_lines_in_transect, add_alpha_to_transect_parts_not_used_dswt_detection

import numpy as np
import matplotlib.pyplot as plt

transects_file = f'input/transects/cwa_transects.json'
    
config = read_config('cwa')

lon_range = [114.0, 116.0]
lat_range = [-33.0, -31.0]

roms_ds = load_roms_data(f'{get_dir_from_json("cwa")}2017/cwa_20170514_03__his.nc', f'{get_dir_from_json("cwa")}grid.nc')
roms_ds = select_roms_subset(roms_ds, None, lon_range, lat_range)

transects = get_transects_in_lon_lat_range(transects_file, lon_range, lat_range)
transect_names = list(transects.keys())
transect_name = transect_names[0]

lon_land = transects[transect_name]['lon_land']
lat_land = transects[transect_name]['lat_land']
lon_ocean = transects[transect_name]['lon_ocean']
lat_ocean = transects[transect_name]['lat_ocean']

transect_ds = select_roms_transect(roms_ds, lon_land, lat_land, lon_ocean, lat_ocean)
l_dswt, _, _, _, _ = determine_dswt_along_transect(transect_ds, config)
vel_down = get_down_transect_velocity_component(transect_ds.u_eastward.values, transect_ds.v_northward.values, lon_land, lat_land, lon_ocean, lat_ocean)
transect_ds['vel_down'] = (['ocean_time', 's_rho', 'distance'], vel_down)

fig = plt.figure(figsize=(8, 12))
ax1 = plt.subplot(3, 1, 1)
ax1, c1 = plot_transect(ax1, transect_ds, 'temp', 0, 20, 22, 'RdYlBu_r')
cbar1 = plt.colorbar(c1)
cbar1.set_label('Temperature')
ax1 = plot_vertical_lines_in_transect(ax1, transect_ds, 'vertical_density_gradient', 0, 0.03)
ax1 = add_alpha_to_transect_parts_not_used_dswt_detection(ax1, transect_ds, config)

ax2 = plt.subplot(3, 1, 2)
ax2, c2 = plot_transect(ax2, transect_ds, 'vertical_density_gradient', 0, 0, 0.03, 'bone_r')
cbar2 = plt.colorbar(c2)
cbar2.set_label('Vertical density gradient')
ax2 = plot_vertical_lines_in_transect(ax2, transect_ds, 'vertical_density_gradient', 0, 0.03, color='w')
ax2 = add_alpha_to_transect_parts_not_used_dswt_detection(ax2, transect_ds, config)

ax3 = plt.subplot(3, 1, 3)
ax3, c3 = plot_transect(ax3, transect_ds, 'vel_down', 0, -0.2, 0.2, 'RdYlBu_r')
cbar3 = plt.colorbar(c3)
cbar3.set_label('Down-transect velocity')
ax3 = plot_vertical_lines_in_transect(ax3, transect_ds, 'vertical_density_gradient', 0, 0.03)
ax3 = add_alpha_to_transect_parts_not_used_dswt_detection(ax3, transect_ds, config)

plt.show()


dswt_cross_transport = calculate_dswt_cross_shelf_transport_along_transect(transect_ds, transects[transect_name], l_dswt, config)
