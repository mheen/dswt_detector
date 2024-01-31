import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from plot_tools.basic_maps import plot_basic_map, plot_contours
from plot_tools.general import add_subtitle

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import xarray as xr
import numpy as np
import pandas as pd

def plot_map(ax:plt.axes, roms_ds:xr.Dataset, transect_ds:xr.Dataset, values:np.ndarray,
             lon_range:list, lat_range:list, meridians:np.ndarray, parallels:np.ndarray,
             vmin:float, vmax:float, cmap:str) -> tuple:
    
    ax = plot_basic_map(ax, lon_range=lon_range, lat_range=lat_range,
                         meridians=meridians, parallels=parallels)
    ax = plot_contours(roms_ds.lon_rho.values, roms_ds.lat_rho.values, roms_ds.h.values,
                        lon_range=lon_range, lat_range=lat_range, ax=ax, clevels=[100, 200])
    c = ax.pcolormesh(roms_ds.lon_rho.values, roms_ds.lat_rho.values, values,
                        vmin=vmin, vmax=vmax, cmap=cmap)
    ax.plot(transect_ds.lon_rho.values, transect_ds.lat_rho.values, '-k')
    
    return ax, c

def plot_transect(ax:plt.axes, transect_ds:xr.Dataset, variable:str, t_dswt:int,
                  vmin:float, vmax:float, cmap:str) -> tuple:
    c = transect_ds[variable][t_dswt, :, :].plot(x='distance', y='z_rho', vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=False)
    ax.fill_between(transect_ds.distance.values, -210, -transect_ds.h.values, color='#989898', edgecolor='k')
    ax.set_xlim([transect_ds.distance.values[0], transect_ds.distance.values[-1]])
    ax.set_ylim([-200, 0])
    ax.set_title('')
    ax.set_ylabel('Depth (m)')
    ax.set_yticks([0, -100, -200])
    ax.set_yticklabels([0, 100, 200])
    
    return ax, c

def base_plot(roms_ds:xr.Dataset, transect_ds:xr.Dataset,
              lon_range:list, lat_range:list,
              parallels:np.ndarray, meridians:np.ndarray,
              t_dswt=0,
              vmin_density=1024.8, vmax_density=1025.4, cmap_density='RdYlBu_r',
              vmin_temp=20., vmax_temp=22., cmap_temp='RdYlBu_r',
              vmin_salt=34.8, vmax_salt=35.8, cmap_salt='RdYlBu_r',
              vmin_drhodz=0.0, vmax_drhodz=0.05, cmap_drhodz='RdYlBu_r') -> plt.figure:
    
    fig = plt.figure(figsize=(5, 9))
    plt.subplots_adjust(hspace=0.2)
    # --- Surface density map
    ax1 = plt.subplot(6, 2, (1, 3), projection=ccrs.PlateCarree())
    ax1, _ = plot_map(ax1, roms_ds, transect_ds, roms_ds.density.values[t_dswt, -1, :, :],
                      lon_range, lat_range, parallels, meridians,
                      vmin_density, vmax_density, cmap_density)
    ax1 = add_subtitle(ax1, '(a) Surface density')
    
    # --- Bottom density map
    ax2 = plt.subplot(6, 2, (2, 4), projection=ccrs.PlateCarree())
    ax2, _ = plot_map(ax2, roms_ds, transect_ds, roms_ds.density.values[t_dswt, 0, :, :],
                      lon_range, lat_range, parallels, meridians,
                      vmin_density, vmax_density, cmap_density)
    ax2 = add_subtitle(ax2, '(b) Bottom density')
    ax2.set_yticklabels([])
    
    # --- Transect density
    ax3 = plt.subplot(6, 2, (5, 6))
    ax3, c3 = plot_transect(ax3, transect_ds, 'density', t_dswt,
                            vmin_density, vmax_density, cmap_density)
    ax3.set_xticks([])
    ax3.set_xlabel('')
    ax3 = add_subtitle(ax3, f'(c) Density along transect', location='lower left')
    
    # adjust location and add colorbar
    l2, b2, w2, h2 = ax2.get_position().bounds
    l3, b3, w3, h3 = ax3.get_position().bounds
    ax3.set_position([l3, b3-0.02, w3, h3])
    cbax3 = fig.add_axes([l3+w3+0.02, b3-0.02, 0.02, h3+h2+0.04])
    cbar3 = plt.colorbar(c3, cax=cbax3)
    cbar3.set_label('Density (kg/m$^3$)')
    
    # --- Transect temperature
    ax4 = plt.subplot(6, 2, (7, 8))
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
    ax5 = plt.subplot(6, 2, (9, 10))
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
    
    # --- Transect vertical density gradient
    ax6 = plt.subplot(6, 2, (11, 12))
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
    
    return fig

def plot_dswt_check(roms_ds:xr.Dataset, transect_ds:xr.Dataset,
                    lon_range:list, lat_range:list,
                    parallels:np.array, meridians:np.array):
    fig = base_plot(roms_ds, transect_ds, lon_range, lat_range, parallels, meridians)
    plt.show()

def plot_dswt_scenario(roms_ds:xr.Dataset, transect_ds:xr.Dataset, l_dswt:list[bool],
                       lon_range:list, lat_range:list,
                       parallels:np.array, meridians:np.array,
                       wind_vel:float, wind_dir:float, t_dswt=0,
                       vmin_density=1024.8, vmax_density=1025.4, cmap_density='RdYlBu_r',
                       vmin_temp=20., vmax_temp=22., cmap_temp='RdYlBu_r',
                       vmin_salt=34.8, vmax_salt=35.8, cmap_salt='RdYlBu_r',
                       vmin_drhodz=0.0, vmax_drhodz=0.05, cmap_drhodz='RdYlBu_r',
                       output_path=None, show=True):
    
    fig = base_plot(roms_ds, transect_ds, t_dswt,
                    lon_range, lat_range, parallels, meridians,
                    vmin_density=vmin_density, vmax_density=vmax_density, cmap_density=cmap_density,
                    vmin_temp=vmin_temp, vmax_temp=vmax_temp, cmap_temp=cmap_temp,
                    vmin_salt=vmin_salt, vmax_salt=vmax_salt, cmap_salt=cmap_salt,
                    vmin_drhodz=vmin_drhodz, vmax_drhodz=vmax_drhodz, cmap_drhodz=cmap_drhodz)
    
    # wind arrow
    ax1 = fig.axes[0]
    l1, b1, w1, h1 = ax1.get_position().bounds
    ax7 = fig.add_axes([l1+0.07, b1+0.5*h1, w1/5, h1/5])
    ax7.text(0, 0, f'{np.round(wind_vel, 0)} m/s', rotation=270-wind_dir, bbox=dict(boxstyle='rarrow', fc='#b0cbb2', ec='k'), fontsize=8)
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