import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from plot_tools.basic_maps import plot_basic_map, plot_contours
from plot_tools.general import add_subtitle

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean as cm

import xarray as xr
import numpy as np
import pandas as pd
import json

cmap_density = cm.cm.thermal_r
cmap_temp = 'RdYlBu_r'
cmap_salt = cm.cm.haline

vmin_drhodz = 0.0
vmax_drhodz = 0.02
cmap_drhodz = 'bone_r'

wind_color = '#f1f5f9'
land_color = '#d2d2d2'

with open('input/plot_settings.json', 'r') as f:
    all_plot_ranges = json.load(f)

def plot_map(ax:plt.axes, roms_ds:xr.Dataset, transect_ds:xr.Dataset, values:np.ndarray,
             lon_range:list, lat_range:list, meridians:np.ndarray, parallels:np.ndarray,
             vmin:float, vmax:float, cmap:str, transect_color='k') -> tuple:
    
    ax = plot_basic_map(ax, lon_range=lon_range, lat_range=lat_range,
                         meridians=meridians, parallels=parallels)
    ax = plot_contours(roms_ds.lon_rho.values, roms_ds.lat_rho.values, roms_ds.h.values,
                        lon_range=lon_range, lat_range=lat_range, ax=ax, clevels=[100, 200])
    c = ax.pcolormesh(roms_ds.lon_rho.values, roms_ds.lat_rho.values, values,
                        vmin=vmin, vmax=vmax, cmap=cmap)
    ax.plot(transect_ds.lon_rho.values, transect_ds.lat_rho.values, '-', color=transect_color)
    
    return ax, c

def plot_transect(ax:plt.axes, transect_ds:xr.Dataset, variable:str, t_dswt:int,
                  vmin:float, vmax:float, cmap:str) -> tuple:
    c = transect_ds[variable][t_dswt, :, :].plot(x='distance', y='z_rho', vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=False)
    ax.fill_between(transect_ds.distance.values, -210, -transect_ds.h.values, color=land_color, edgecolor='k')
    ax.set_xlim([transect_ds.distance.values[0], transect_ds.distance.values[-1]])
    ax.set_ylim([-200, 0])
    ax.set_title('')
    ax.set_ylabel('Depth (m)')
    ax.set_yticks([0, -100, -200])
    ax.set_yticklabels([0, 100, 200])
    
    return ax, c

def transects_plot(transect_ds:xr.Dataset, t_dswt:int,
                   fig:plt.figure, n_rows:int, n_cols:int, n_start:int,
                   set_vlim=True) -> plt.figure:
    
    if set_vlim == True:
        time_str = pd.to_datetime(transect_ds.ocean_time.values[0]).strftime('%b')
        vmin_density = all_plot_ranges[time_str]['vmin_density']
        vmax_density = all_plot_ranges[time_str]['vmax_density']
        vmin_temp = all_plot_ranges[time_str]['vmin_temp']
        vmax_temp = all_plot_ranges[time_str]['vmax_temp']
        vmin_salt = all_plot_ranges[time_str]['vmin_salt']
        vmax_salt = all_plot_ranges[time_str]['vmax_salt']
    else:
        vmin_density = None
        vmax_density = None
        vmin_temp = None
        vmax_temp = None
        vmin_salt = None
        vmax_salt = None
      
    # --- Transect density
    ax3 = plt.subplot(n_rows, n_cols, (n_start, n_start+1))
    ax3, c3 = plot_transect(ax3, transect_ds, 'density', t_dswt,
                            vmin_density, vmax_density, cmap_density)
    ax3.set_xticks([])
    ax3.set_xlabel('')
    ax3 = add_subtitle(ax3, f'(c) Density along transect', location='lower left')
    
    # adjust location and add colorbar
    l3, b3, w3, h3 = ax3.get_position().bounds
    ax3.set_position([l3, b3-0.02, w3, h3])
    cbax3 = fig.add_axes([l3+w3+0.02, b3-0.02, 0.02, h3])
    cbar3 = plt.colorbar(c3, cax=cbax3)
    cbar3.set_label('Density (kg/m$^3$)')
    
    # --- Transect temperature
    ax4 = plt.subplot(n_rows, n_cols, (n_start+2, n_start+3))
    ax4, c4 = plot_transect(ax4, transect_ds, 'temp', t_dswt,
                            vmin_temp, vmax_temp, cmap_temp)
    ax4.set_xlabel('')
    ax4.set_xticks([])
    
    ax4 = add_subtitle(ax4, f'(d) Temperature along transect', location='lower left')
    
    # adjust location and add colorbar
    l4, b4, w4, h4 = ax4.get_position().bounds
    ax4.set_position([l4, b4-0.02, w4, h4])
    cbax4 = fig.add_axes([l4+w4+0.02, b4-0.02, 0.02, h4])
    cbar4 = plt.colorbar(c4, cax=cbax4)
    cbar4.set_label('Temperature ($^o$C)', fontsize=8)
    
    # --- Transect salinity
    ax5 = plt.subplot(n_rows, n_cols, (n_start+4, n_start+5))
    ax5, c5 = plot_transect(ax5, transect_ds, 'salt', t_dswt,
                            vmin_salt, vmax_salt, cmap_salt)
    ax5.set_xlabel('')
    ax5.set_xticks([])
    ax5 = add_subtitle(ax5, f'(e) Salinity along transect', location='lower left')
    
    # adjust location and add colorbar
    l5, b5, w5, h5 = ax5.get_position().bounds
    ax5.set_position([l5, b5-0.02, w5, h5])
    cbax5 = fig.add_axes([l5+w5+0.02, b5-0.02, 0.02, h5])
    cbar5 = plt.colorbar(c5, cax=cbax5)
    cbar5.set_label('Salinity (ppt)', fontsize=8)
    
    # title
    plt.suptitle(f'{pd.to_datetime(transect_ds.ocean_time[t_dswt].values).strftime("%d-%m-%Y %H:%M")}', x=0.5, y=0.93)    
    
    return fig

def plot_dswt_check(transect_ds:xr.Dataset, t_dswt:int):
    
    fig = plt.figure(figsize=(6, 8))
    n_rows = 3
    n_cols = 2
    n_start = 1
    fig = transects_plot(transect_ds, t_dswt, fig, n_rows, n_cols, n_start, set_vlim=False)
    plt.show()

def plot_dswt_scenario(roms_ds:xr.Dataset, transect_ds:xr.Dataset, l_dswt:list[bool], t_dswt:int,
                       wind_vel:float, wind_dir:float,
                       lon_range:list, lat_range:list,
                       parallels:np.array, meridians:np.array,
                       output_path=None, show=True):
    
    time_str = pd.to_datetime(transect_ds.ocean_time.values[0]).strftime('%b')
    vmin_density = all_plot_ranges[time_str]['vmin_density']
    vmax_density = all_plot_ranges[time_str]['vmax_density']
    
    fig = plt.figure(figsize=(5, 8))
    plt.subplots_adjust(hspace=0.2)
    
    n_rows = 6
    n_cols = 2
    
    # --- Surface density map
    ax1 = plt.subplot(n_rows, n_cols, (1, 3), projection=ccrs.PlateCarree())
    ax1, _ = plot_map(ax1, roms_ds, transect_ds, roms_ds.density.values[t_dswt, -1, :, :],
                      lon_range, lat_range, parallels, meridians,
                      vmin_density, vmax_density, cmap_density)
    ax1 = add_subtitle(ax1, '(a) Surface density')
    
    # --- Bottom density map
    ax2 = plt.subplot(6, 2, (2, 4), projection=ccrs.PlateCarree())
    ax2, _ = plot_map(ax2, roms_ds, transect_ds, roms_ds.density.values[t_dswt, 0, :, :],
                      lon_range, lat_range, parallels, meridians,
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
    ax7.text(0, 0, f'{np.round(wind_vel, 0)} m/s', rotation=270-wind_dir, bbox=dict(boxstyle='rarrow', fc=wind_color, ec='k'), fontsize=8)
    ax7.set_axis_off()
    
    # title
    dswt_str = 'DSWT' if l_dswt[t_dswt] == True else 'no DSWT'
    plt.suptitle(f'{pd.to_datetime(roms_ds.ocean_time[0].values).strftime("%d-%m-%Y %H:%M")} - {dswt_str}', x=0.5, y=0.93)
    
    if output_path is not None:
        # save figure
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    if show is True:
        plt.show()
    else:
        plt.close()