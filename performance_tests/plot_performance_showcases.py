import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from readers.read_ocean_data import load_roms_data, select_roms_transect_from_known_coordinates
from readers.read_meteo_data import load_era5_data, select_era5_in_closest_point, get_daily_mean_wind_data
from dswt.dswt_detection import determine_dswt_along_transect
from dswt.cross_shelf_transport import calculate_dswt_cross_shelf_transport_along_transect, add_down_transect_velocity_to_ds
from transects import read_transects_in_lon_lat_range_from_json
from plot_tools.dswt import plot_map, transects_plot, plot_transect
from plot_tools.dswt import plot_vertical_lines_in_transect, add_alpha_to_transect_parts_not_used_dswt_detection
from plot_tools.general import add_subtitle
from tools.files import get_dir_from_json, create_dir_if_does_not_exist
from tools.config import read_config, Config
from datetime import datetime
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean as cm
import xarray as xr

year = 2017

transects_file = 'input/transects/cwa_transects.json'
transects = read_transects_in_lon_lat_range_from_json(transects_file, [114.0, 116.0], [-33.0, -31.0])

main_input_dir = get_dir_from_json('cwa')
input_dir = f'{main_input_dir}{year}/'
grid_file = f'{main_input_dir}grid.nc'

output_dir = 'performance_tests/plots/'
create_dir_if_does_not_exist(output_dir)

plot_lon_range = [114.5, 115.8]
plot_lat_range = [-33.0, -31.5]
meridians = [114.5, 115.5]
parallels = [-33.0, -32.0, -31.0]

cmap_density = cm.cm.thermal_r
cmap_temp = 'RdYlBu_r'
cmap_salt = cm.cm.haline

vmin_drhodz = 0.0
vmax_drhodz = 0.02
cmap_drhodz = 'bone_r'

wind_color = '#f1f5f9'
land_color = '#d2d2d2'

def plot_dswt_detection(date:datetime, transect_to_plot:int, time_to_plot:int, config:Config,
                        output_path=None):  
    # -- Load ROMS data
    input_path = f'{input_dir}cwa_{date.strftime("%Y%m%d")}_03__his.nc'
    roms_ds = load_roms_data(input_path, grid_file=grid_file)
    
    eta = transects[transect_to_plot]['eta']
    xi = transects[transect_to_plot]['xi']
    transect_ds = select_roms_transect_from_known_coordinates(roms_ds, eta, xi)
    l_dswt, _, _, _, _ = determine_dswt_along_transect(transect_ds, config)
    
    # --- Load ERA5 wind data
    era5_ds = load_era5_data(get_dir_from_json("era5"), '2017')
    lon_p = transect_ds.lon_rho.values[np.floor(len(transect_ds.lon_rho)/2).astype(int)]
    lat_p = transect_ds.lat_rho.values[np.floor(len(transect_ds.lat_rho)/2).astype(int)]
    point_ds = select_era5_in_closest_point(era5_ds, lon_p, lat_p)
    time, _, _, wind_vel, wind_dir = get_daily_mean_wind_data(point_ds)
    i_wind = np.where(time == date)[0][0]
    
    with open('input/plot_settings.json', 'r') as f:
        all_plot_ranges = json.load(f)
    
    time_str = pd.to_datetime(transect_ds.ocean_time.values[0]).strftime('%b')
    vmin_density = all_plot_ranges[time_str]['vmin_density']
    vmax_density = all_plot_ranges[time_str]['vmax_density']
    
    fig = plt.figure(figsize=(5, 8))
    plt.subplots_adjust(hspace=0.2)
    
    n_rows = 6
    n_cols = 2
    
    t_dswt = time_to_plot
    
    # --- Surface density map
    ax1 = plt.subplot(n_rows, n_cols, (1, 3), projection=ccrs.PlateCarree())
    ax1, _ = plot_map(ax1, roms_ds, transect_ds, roms_ds.density.values[t_dswt, -1, :, :],
                    plot_lon_range, plot_lat_range, parallels, meridians,
                    vmin_density, vmax_density, cmap_density)
    ax1 = add_subtitle(ax1, '(a) Surface density')
    
    # --- Bottom density map
    ax2 = plt.subplot(6, 2, (2, 4), projection=ccrs.PlateCarree())
    ax2, _ = plot_map(ax2, roms_ds, transect_ds, roms_ds.density.values[t_dswt, 0, :, :],
                    plot_lon_range, plot_lat_range, parallels, meridians,
                    vmin_density, vmax_density, cmap_density, transect_color='#cccccc')
    ax2 = add_subtitle(ax2, '(b) Bottom density')
    ax2.set_yticklabels([])
    
    fig = transects_plot(transect_ds, t_dswt, fig, n_rows, n_cols, 5)
    
    # --- Transect vertical density gradient
    ax6 = plt.subplot(n_rows, n_cols, (11, 12))
    ax6, c6 = plot_transect(ax6, transect_ds, 'vertical_density_gradient', t_dswt,
                            vmin_drhodz, vmax_drhodz, cmap_drhodz)
    ax6.set_xlabel('Distance (m)')
    ax6 = add_subtitle(ax6, f'(f) Vertical density gradient along transect', location='lower left')
    
    # adjust location and add colorbar
    l6, b6, w6, h6 = ax6.get_position().bounds
    ax6.set_position([l6, b6-0.02, w6, h6])
    cbax6 = fig.add_axes([l6+w6+0.02, b6-0.02, 0.02, h6])
    cbar6 = plt.colorbar(c6, cax=cbax6)
    cbar6.set_label('Vertical\ndensity gradient\n(kg/m$^3$/m)', fontsize=8)
    
    # wind arrow
    ax1 = fig.axes[0]
    l1, b1, w1, h1 = ax1.get_position().bounds
    ax7 = fig.add_axes([l1+0.07, b1+0.5*h1, w1/5, h1/5])
    ax7.text(0, 0, f'{np.round(wind_vel[i_wind], 0)} m/s', rotation=270-wind_dir[i_wind], bbox=dict(boxstyle='rarrow', fc=wind_color, ec='k'), fontsize=8)
    ax7.set_axis_off()
    
    # title
    dswt_str = 'DSWT' if l_dswt[t_dswt] == True else 'no DSWT'
    plt.suptitle(f'{pd.to_datetime(roms_ds.ocean_time[0].values).strftime("%d-%m-%Y %H:%M")} - {dswt_str}', x=0.5, y=0.93)
    
    if output_path is not None:
        # save figure
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

def plot_dswt_cross_shelf_transport(mean_transect_ds:xr.Dataset,
                                    transport:list[float],
                                    velocity:list[float],
                                    dz:list[float],
                                    ds:list[float],
                                    h:list[float],
                                    title='',
                                    output_path=None,
                                    show=True):
    
    def _plot_detection_boxes(ax:plt.axes, i):
        # distances are not even: taking mean as approximation
        dx = np.nanmean(np.diff(transect_ds.distance.values))
        x = [ds[i]-dx, ds[i]+dx, ds[i]+dx, ds[i]-dx, ds[i]-dx]
        depth0 = -h[i]
        depth1 = -h[i]+dz[i]
        y = [depth0, depth0, depth1, depth1, depth0]
        ax.plot(x, y, '-r', linewidth=2)

    fig = plt.figure(figsize=(8, 12))
    ax1 = plt.subplot(3, 1, 1)
    ax1, c1 = plot_transect(ax1, mean_transect_ds, 'density', 0, 1024.8, 1025.4, cmap_density)
    cbar1 = plt.colorbar(c1)
    cbar1.set_label('Density (kg/m$^3$)')
    ax1 = plot_vertical_lines_in_transect(ax1, mean_transect_ds, 'vertical_density_gradient', 0, 0.03)
    for i in range(len(ds)):
        _plot_detection_boxes(ax1, i)
        
    ax1 = add_subtitle(ax1, '(a) Density along transect', location='lower left')

    ax2 = plt.subplot(3, 1, 2)
    ax2, c2 = plot_transect(ax2, mean_transect_ds, 'vertical_density_gradient', 0, 0, 0.03, 'bone_r')
    cbar2 = plt.colorbar(c2)
    cbar2.set_label('Vertical density gradient')
    ax2 = plot_vertical_lines_in_transect(ax2, mean_transect_ds, 'vertical_density_gradient', 0, 0.03, color='#808080')
    for i in range(len(ds)):
        _plot_detection_boxes(ax2, i)
        
    ax2 = add_subtitle(ax2, '(b) Vertical density gradient along transect', location='lower left')

    ax3 = plt.subplot(3, 1, 3)
    ax3, c3 = plot_transect(ax3, mean_transect_ds, 'down_transect_vel', 0, -0.2, 0.2, 'RdYlBu_r')
    cbar3 = plt.colorbar(c3)
    cbar3.set_label('Down-transect velocity')
    ax3 = plot_vertical_lines_in_transect(ax3, mean_transect_ds, 'vertical_density_gradient', 0, 0.03)
    for i in range(len(ds)):
        _plot_detection_boxes(ax3, i)
        
    cross_text = f'Mean velocity {np.round(np.nanmean(velocity), 2)} m/s, total transport: {np.round(np.nansum(transport), 0).astype(int)} m$^2$/day'
    ax3 = add_subtitle(ax3, f'(c) Cross-shelf velocity along transect\n    {cross_text}',
                       location='lower left')

    ax1.set_title(title)

    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    if show is True:
        plt.show()
    else:
        plt.close()

if __name__ == '__main__':
    make_dswt_detection_plots = False
    make_cross_transport_plots = True
    
    if make_dswt_detection_plots == True:
        # ------------------------------------------------------
        # DSWT detection examples
        # ------------------------------------------------------
        dates = [datetime(2017, 1, 15), # no DSWT: positive drhodx
                datetime(2017, 2, 22), # no DSWT: negative drhodx because of higher salinity along coast, but vertically mixed water column
                datetime(2017, 6, 11), # DSWT
                datetime(2017, 9, 8), # manual DSWT, algorithm no DSWT (because not enough consecutive cells)
                datetime(2017, 5, 21),  # manual no DSWT, algorithm DSWT
                datetime(2017, 8, 19),
                datetime(2017, 8, 18)]
        transects_to_plot = ['t201', 't197', 't171', 't209', 't188', 't268', 't200']
        time_to_plot = [0, 0, 6, 0, 5, 0, 3]

        config = read_config('cwa')

        for i in range(len(dates)):
            output_path = f'{output_dir}cwa_dswt-detection_{dates[i].strftime("%Y%m%d")}_{transects_to_plot[i]}.jpg'
            plot_dswt_detection(dates[i], transects_to_plot[i], time_to_plot[i], config, output_path=output_path)

    if make_cross_transport_plots == True:
        # ------------------------------------------------------
        # Cross-shelf transport examples
        # ------------------------------------------------------
        # dates = [datetime(2017, 5, 14), datetime(2017, 8, 19), datetime(2017, 8, 18)]
        # transects_to_plot = ['t135', 't268', 't200']
        # time_to_plot = [3, 5, 6]
        dates = [datetime(2017, 4, 20)]
        transects_to_plot = ['t153']
        
        config = read_config('cwa')
        
        for i in range(len(dates)):
            output_path = f'{output_dir}cwa_cross-transport_{dates[i].strftime("%Y%m%d")}_{transects_to_plot[i]}.jpg'
            # -- Load ROMS data
            input_path = f'{input_dir}cwa_{dates[i].strftime("%Y%m%d")}_03__his.nc'
            roms_ds = load_roms_data(input_path, grid_file=grid_file)
            
            eta = transects[transects_to_plot[i]]['eta']
            xi = transects[transects_to_plot[i]]['xi']
            transect_ds = select_roms_transect_from_known_coordinates(roms_ds, eta, xi)
            transect_ds = add_down_transect_velocity_to_ds(transect_ds)
            l_dswt, _, _, _, _ = determine_dswt_along_transect(transect_ds, config)
            cross_transport, cross_vel, dz, ds, _, _, h = calculate_dswt_cross_shelf_transport_along_transect(transect_ds, l_dswt, config)
            
            time_str = f'{pd.to_datetime(transect_ds.ocean_time.values[0]).strftime("%d-%m-%Y")}'
            title = f'DSWT cross-shelf transport: {transects_to_plot[i]}, {time_str}'
            
            mean_transect_ds = transect_ds.isel(ocean_time=l_dswt).mean(dim='ocean_time', keepdims=True) # mean over times with DSWT
            
            plot_dswt_cross_shelf_transport(transect_ds, cross_transport, cross_vel, dz, ds, h,
                                            title=title, output_path=output_path, show=False)
