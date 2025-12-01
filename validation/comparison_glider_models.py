import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from validation.glider_data import GliderData
from validation.glider_dswt import load_glider_transect_data, determine_dswt_along_glider_transect
from readers.read_ocean_data import load_roms_data, select_roms_transect_from_known_coordinates, select_roms_transect_from_start_end_coordinates
from dswt.dswt_detection import determine_dswt_along_transect
from tools.roms import get_eta_xi_of_lon_lat_point
from tools.files import get_dir_from_json, get_files_in_dir, get_daily_files_in_time_range
from tools.config import read_config, Config
from tools.timeseries import get_l_time_range
from tools.coordinates import get_unique_coordinates
import xarray as xr
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
from plot_tools.basic_maps import plot_basic_map
import cartopy.crs as ccrs
import os
import cmocean as cm


lon_range_default = [114.5, 116.]
lat_range_default = [-33., -31.]
meridians_default = [115., 116.]
parallels_default = [-33., -32., -31.]

def _get_vmin_vmax(glider_d:np.ndarray[float]):
    bins = np.arange(1024.5, 1026.6, 0.1)
    bin_edges = np.empty(len(bins)+1)
    bin_edges[:-1] = bins - 0.05
    bin_edges[-1] = bins[-1] + 0.05
    
    n, _ = np.histogram(glider_d[~np.isnan(glider_d)], bins=bin_edges)
    bins_most_values = bins[n >= 0.2 * np.nanmax(n)]
    
    vmin = bins_most_values[0]
    vmax = bins_most_values[-1]
    return vmin, vmax

def compare_glider_model_dswt_detection(glider_dir=get_dir_from_json('glider_transects'),
                                        model_dir=get_dir_from_json('cwa')):
    glider_files = get_files_in_dir(glider_dir, 'nc')
    
    grid_file = f'{model_dir}grid.nc'
    
    output_path='validation/output/comparison.csv'
    
    config = read_config('cwa')
    
    df = pd.DataFrame(index=np.arange(0, 2*len(glider_files)),
                      columns=['transect', 'type', 'vel', 'thickness', 'transport', 'distance', 'h', 'h_max',
                               'mean_drhodx', 'min_drhodx', 'mean_drhos', 'min_drhos'])
    
    for i, glider_file in enumerate(glider_files):
        transect_name = os.path.splitext(os.path.basename(glider_file))[0]
        
        glider_transect_ds = load_glider_transect_data(glider_file)
        start_date = pd.to_datetime(glider_transect_ds.time.values[0])
        end_date = pd.to_datetime(glider_transect_ds.time.values[-1])
        lon0 = glider_transect_ds.lon.values[0]
        lat0 = glider_transect_ds.lat.values[0]
        lon1 = glider_transect_ds.lon.values[-1]
        lat1 = glider_transect_ds.lat.values[-1]
        
        model_files = get_daily_files_in_time_range(f'{model_dir}{start_date.year}/', start_date, end_date, 'nc')
        roms_ds = load_roms_data(model_files, grid_file)
        transect_ds = select_roms_transect_from_start_end_coordinates(roms_ds, lon0, lat0, lon1, lat1)
        l_time = get_l_time_range(transect_ds.ocean_time.values, start_date, end_date)
        transect_ds = transect_ds.isel(ocean_time=np.where(l_time)[0])
        
        (glider_vel, glider_thickness, glider_transport, glider_distance, _, _, glider_h, 
        glider_drhodx, glider_mean_drhos,  glider_min_drhos) = determine_dswt_along_glider_transect(glider_transect_ds, config)
        (t_dswt, _, vel, thickness, transport, distance, _, _, h, mean_drhodx, min_drhodx, mean_drhos, min_drhos) = determine_dswt_along_transect(transect_ds, config, mld_condition=False)
        
        plot_comparison(glider_transect_ds, transect_ds, transect_name, glider_distance, -glider_h + glider_thickness,
                        distance, -h + thickness, t_dswt)
        
        df.loc[2*i] = [transect_name, 'glider', np.nanmean(glider_vel), np.nanmean(glider_thickness), np.nanmean(glider_transport),
                       np.nanmean(glider_distance), np.nanmean(glider_h), nanmaxempty(glider_h),
                        glider_drhodx, glider_drhodx, np.nanmean(glider_mean_drhos), nanminempty(glider_min_drhos)]
        df.loc[2*i+1] = [transect_name, 'model', np.nanmean(vel), np.nanmean(thickness), np.nanmean(transport), np.nanmean(distance),
                         np.nanmean(h), nanmaxempty(h), mean_drhodx, min_drhodx, np.nanmean(mean_drhos), nanminempty(min_drhos)]
    df = df.sort_values(by='transect')
    df.to_csv(output_path, index=False)

def plot_comparison(glider_transect_ds, transect_ds, transect_name,
                    x_glider, z_glider, x_model, z_model, t_dswt, cmap=cm.cm.thermal_r):
    glider_d = glider_transect_ds.density.values
    vmin, vmax = _get_vmin_vmax(glider_d)
    
    fig = plt.figure(figsize=(8, 5))
    
    # glider transect
    glider_x = glider_transect_ds.distance.values
    glider_z = glider_transect_ds.z.values
    glider_h = glider_transect_ds.h.values
    ax1 = plt.subplot(2, 1, 1)
    c1 = ax1.pcolormesh(glider_x, glider_z, glider_d, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.scatter(x_glider, z_glider, marker='x', c='w', s=30)
    ax1.scatter(x_glider, z_glider, marker='x', c='k', s=20)
    ax1.fill_between(glider_x, -110, glider_h, color='#d2d2d2', edgecolor='k')
    ax1.set_ylim([-100, 0])
    ax1.set_ylabel('Depth (m)')
    ax1.set_yticks([0, -25, -50, -75, -100])
    ax1.set_yticklabels([0, 25, 50, 75, 100])
    ax1.set_xlim([glider_x[0], glider_x[-1]])
    
    # model transect
    x = transect_ds.distance.values
    z = transect_ds.z_rho.values
    if len(t_dswt) != 0:
        d = np.nanmean(transect_ds.density.values[t_dswt, :, :], axis=0)
    else:
        d = np.nanmean(transect_ds.density.values, axis=0)
    h = -transect_ds.h.values
    ax2 = plt.subplot(2, 1, 2)
    c2 = ax2.pcolormesh(x, z, d, cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.scatter(x_model, z_model, marker='x', c='w', s=30)
    ax2.scatter(x_model, z_model, marker='x', c='k', s=20)
    ax2.fill_between(x, -110, h, color='#d2d2d2', edgecolor='k')
    ax2.set_ylim([-100, 0])
    ax2.set_ylabel('Depth (m)')
    ax2.set_yticks([0, -25, -50, -75, -100])
    ax2.set_yticklabels([0, 25, 50, 75, 100])
    ax2.set_xlim([glider_x[0], glider_x[-1]])
    ax2.set_xlabel('Distance along transect (m)')
    
    # colorbar
    ll, bb, ww, hh = ax2.get_position().bounds
    cax = fig.add_axes([ll, bb-0.16, ww, 0.04])
    cbar = plt.colorbar(c2, cax=cax, orientation='horizontal')
    cbar.set_label('Density (kg m$^{-3}$)')
    
    # maps with transect
    l1, b1, w1, h1 = ax1.get_position().bounds
    axm1 = fig.add_axes([l1+0.005, b1+0.02, w1/4, h1/2], projection=ccrs.PlateCarree())
    plot_basic_map(axm1, [115.0, 116.0], [-32.5, -31.5])
    axm1.plot(glider_transect_ds.lon.values, glider_transect_ds.lat.values, '-', color='#C70039', linewidth=1)
    axm1.set_xticks([])
    axm1.set_yticks([])
    
    axm2 = fig.add_axes([ll+0.005, bb+0.02, ww/4, hh/2], projection=ccrs.PlateCarree())
    plot_basic_map(axm2, [115.0, 116.0], [-32.5, -31.5])
    axm2.plot(transect_ds.lon_rho.values, transect_ds.lat_rho.values, '-', color='#C70039', linewidth=1)
    axm2.set_xticks([])
    axm2.set_yticks([])
    
    plt.savefig(f'validation/output/{transect_name}.jpg', bbox_inches='tight', dpi=300)

def plot_glider_transect(glider_transect_ds, transect_name, config:Config, cmap=cm.cm.thermal_r):
    (_, dswt_glider_thickness, _, dswt_glider_distance, _, _, dswt_glider_h,
     glider_drhodx, glider_mean_drhos,  glider_min_drhos) = determine_dswt_along_glider_transect(glider_transect_ds, config)
    
    glider_d = glider_transect_ds.density.values
    vmin, vmax = _get_vmin_vmax(glider_d)
    
    fig = plt.figure(figsize=(8, 4))
    
    # glider transect
    glider_x = glider_transect_ds.distance.values
    glider_z = glider_transect_ds.z.values
    glider_h = glider_transect_ds.h.values
    ax1 = plt.subplot(2, 1, 1)
    c1 = ax1.pcolormesh(glider_x, glider_z, glider_d, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.scatter(dswt_glider_distance, -dswt_glider_h + dswt_glider_thickness, marker='x', c='w', s=30)
    ax1.scatter(dswt_glider_distance, -dswt_glider_h + dswt_glider_thickness, marker='x', c='k', s=20)
    ax1.fill_between(glider_x, -110, glider_h, color='#d2d2d2', edgecolor='k')
    ax1.set_ylim([-100, 0])
    ax1.set_ylabel('Depth (m)')
    ax1.set_yticks([0, -25, -50, -75, -100])
    ax1.set_yticklabels([0, 25, 50, 75, 100])
    ax1.set_xlim([glider_x[0], glider_x[-1]])
    
    # colorbar
    l1, b1, w1, h1 = ax1.get_position().bounds
    cax = fig.add_axes([l1, b1-0.16, w1, 0.04])
    cbar = plt.colorbar(c1, cax=cax, orientation='horizontal')
    cbar.set_label('Density (kg m$^{-3}$)')
    
    # maps with transect
    axm1 = fig.add_axes([l1+0.005, b1+0.02, w1/4, h1/2], projection=ccrs.PlateCarree())
    plot_basic_map(axm1, [115.0, 116.0], [-32.5, -31.5])
    axm1.plot(glider_transect_ds.lon.values, glider_transect_ds.lat.values, '-', color='#C70039', linewidth=1)
    axm1.set_xticks([])
    axm1.set_yticks([])
    
    plt.savefig(f'validation/output/glider_{transect_name}.jpg', bbox_inches='tight', dpi=300)

def nanmaxempty(array):
    if len(array) == 0:
        return np.nan
    return np.nanmax(array)

def nanminempty(array):
    if len(array) == 0:
        return np.nan
    return np.nanmin(array)

def plot_glider_model_comparison_measures(input_path='validation/output/comparison.csv'):
    df = pd.read_csv(input_path)
    df_glider = df[df['type'] == 'glider']
    df_model = df[df['type'] == 'model']
    
    fig = plt.figure(figsize=(8, 11))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    def _plot_glider_model_comparison(ax:plt.axes, glider_values:np.array, model_values:np.array):
        ax.plot(glider_values, model_values, 'xk')
        min_value = np.nanmin([glider_values, model_values])
        max_value = np.nanmax([glider_values, model_values])
        dvalue = max_value - min_value
        buffer = 0.1
        ax.plot([min_value-buffer*dvalue, max_value+buffer*dvalue], [min_value-buffer*dvalue, max_value+buffer*dvalue], '--k')
        ax.set_xlim([min_value-buffer*dvalue, max_value+buffer*dvalue])
        ax.set_ylim([min_value-buffer*dvalue, max_value+buffer*dvalue])
        ax.set_xlabel('Glider')
        ax.set_ylabel('Model')

    # thickness        
    ax1 = plt.subplot(3, 2, 1)
    _plot_glider_model_comparison(ax1, df_glider['thickness'].values, df_model['thickness'].values)
    ax1.set_title('DSWT mean thickness (m)')
    
    # mean_drhodx
    ax2 = plt.subplot(3, 2, 2)
    _plot_glider_model_comparison(ax2, df_glider['mean_drhodx'].values, df_model['mean_drhodx'].values)
    ax2.set_title('DSWT mean dp/dx')
    
    # mean_drhos
    ax3 = plt.subplot(3, 2, 3)
    _plot_glider_model_comparison(ax3, df_glider['mean_drhos'].values, df_model['mean_drhos'].values)
    ax3.set_title('DSWT mean dp s / p0')
    
    # min_drhos
    ax4 = plt.subplot(3, 2, 4)
    _plot_glider_model_comparison(ax4, df_glider['min_drhos'].values, df_model['min_drhos'].values)
    ax4.set_title('DSWT min dp s / p0')
    
    # distance (?) maybe not super useful since transect lengths not very comparable?
    ax5 = plt.subplot(3, 2, 5)
    _plot_glider_model_comparison(ax5, df_glider['distance'].values, df_model['distance'].values)
    ax5.set_title('DSWT mean distance (m)')
    
    # h or h_max (?) some bathymetry issues there, so maybe also not the best?
    ax6 = plt.subplot(3, 2, 6)
    _plot_glider_model_comparison(ax6, df_glider['h_max'].values, df_model['h_max'].values)
    ax6.set_title('DSWT max depth (m)')
    
    plt.savefig(f'validation/output/glider_model_comparison.jpg', bbox_inches='tight', dpi=300)
    

if __name__ == '__main__':
    compare_glider_model_dswt_detection()
    plot_glider_model_comparison_measures()
    