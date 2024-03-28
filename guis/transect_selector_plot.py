import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from plot_tools.interactive_tools import plot_cycler
from readers.read_ocean_data import select_roms_transect
from tools.roms import get_eta_xi_along_transect
from dswt.dswt_detection import determine_dswt_along_transect

from tools.config import read_config, Config
from readers.read_ocean_data import load_roms_data
from tools.files import get_dir_from_json
from transects import get_transects_in_lon_lat_range

import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
from datetime import datetime

def select_transects_to_plot(transects:dict, transect_interval:int,
                             lon_rho:np.ndarray, lat_rho:np.ndarray,
                             transect_ds=500.) -> list[dict]:
    
    transect_names = list(transects.keys())
    
    transects_to_plot = []
    for i in np.arange(0, len(transect_names), transect_interval):
        lon_land = transects[transect_names[i]]['lon_land']
        lat_land = transects[transect_names[i]]['lat_land']
        lon_ocean = transects[transect_names[i]]['lon_ocean']
        lat_ocean = transects[transect_names[i]]['lat_ocean']
        
        eta, xi = get_eta_xi_along_transect(lon_rho, lat_rho,
                                            lon_land, lat_land,
                                            lon_ocean, lat_ocean,
                                            transect_ds)
       
        transect = {'name': transect_names[i],
                    'lon':lon_rho[eta, xi], 'lat': lat_rho[eta, xi],
                    'lon_land': lon_land, 'lat_land': lat_land,
                    'lon_ocean': lon_ocean, 'lat_ocean': lat_ocean}

        transects_to_plot.append(transect)
        
    return transects_to_plot

def get_dswt_transect(df_dswt:pd.DataFrame, transect_name:str, time:datetime) -> bool:
    # not ideal: returns True if there is ANY DSWT during a day
    f_dswt = df_dswt.xs(time.strftime("%Y-%m-%d"), level='time', drop_level=False).xs(transect_name, level='transect', drop_level=False)['f_dswt']
    if f_dswt.values[0] > 0.0:
        return True
    else:
        return False

def plot_transects(ax:plt.axes, transects:list[dict], df_dswt:pd.DataFrame, time:datetime) -> tuple[plt.axes, plt.legend]:
    for transect in transects:
        if df_dswt is not None:
            l_dswt = get_dswt_transect(df_dswt, transect['name'], time)
        else:
            l_dswt = None
        if l_dswt == True:
            ax.plot(transect['lon'], transect['lat'], '-', color='#1b7931', label=transect['name'])
        elif l_dswt == False:
            ax.plot(transect['lon'], transect['lat'], '-', color='k', label=transect['name'])
        else:
            ax.plot(transect['lon'], transect['lat'], '-', color='#989898')
    # legend
    legend_elements = [Line2D([0], [0], color='#1b7931'), Line2D([0], [0], color='k'), Line2D([0], [0], color='#989898')]
    l = ax.legend(legend_elements, ['DSWT', 'no DSWT', 'unknown'], loc='upper left', bbox_to_anchor=(1.01, 1.0))
    
    return ax, l

def get_vmin_vmax(vmin:float, vmax:float, variable:str, roms_ds:xr.Dataset) -> tuple[float, float]:
    if vmin is None:
        if variable == 'temp':
            vmin = 20.0
        elif variable == 'density':
            vmin = 1024.8
        elif variable == 'salt':
            vmin = 34.8
        elif variable == 'vertical_density_gradient':
            vmin = -0.005
        else:
            vmin = np.round(np.nanmin(roms_ds[variable].values), 1)
    if vmax is None:
        if variable == 'temp':
            vmax = 22.0
        elif variable == 'density':
            vmax = 1025.4
        elif variable == 'salt':
            vmax = 35.8
        elif variable == 'vertical_density_gradient':
            vmax = 0.05
        else:
            vmax = np.round(np.nanmin(roms_ds[variable].values), 1)
    return vmin, vmax

def plot_dswt_maps_transects(roms_ds:xr.Dataset,
                             transects:dict,
                             df_dswt:pd.DataFrame,
                             config:Config,
                             variable='density', vmin=None, vmax=None, cmap='RdYlBu_r',
                             t_interval=1, transect_interval=4,
                             lon_range=None, lat_range=None) -> plt.axes:

    if variable not in roms_ds.variables:
        raise ValueError(f'''Unknown variable requested: {variable}
                         Valid variables are: {list(roms_ds.keys())}''')
    
    transects_to_plot = select_transects_to_plot(transects, transect_interval,
                                         roms_ds.lon_rho.values, roms_ds.lat_rho.values)
    
    time_strs = pd.to_datetime(roms_ds.ocean_time.values)
    
    vmin, vmax = get_vmin_vmax(vmin, vmax, variable, roms_ds) 
    
    def single_plot(fig, req_time):
        
        # time index to plot
        t = list(roms_ds.ocean_time.values).index(roms_ds.sel(ocean_time=req_time, method='nearest').ocean_time.values)
        
        # --------------------------------------------------------
        # Maps
        # --------------------------------------------------------
        
        # --- Surface map
        ax1 = plt.subplot(3, 2, 1, projection=ccrs.PlateCarree())
        c1 = roms_ds[variable][t, -1, :, :].plot(ax=ax1, transform=ccrs.PlateCarree(), x='lon_rho', y='lat_rho',
                                                  vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=False)
        ax1.set_title('Surface')
        
        # --- Bottom map
        ax2 = plt.subplot(3, 2, 2, projection=ccrs.PlateCarree(), sharex=ax1, sharey=ax1)
        c2 = roms_ds[variable][t, 0, :, :].plot(ax=ax2, transform=ccrs.PlateCarree(), x='lon_rho', y='lat_rho',
                                                 vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=False)
        ax2.set_title('Bottom')
        
        # --- Reposition axes
        l1, b1, w1, h1 = ax1.get_position().bounds
        l2, b2, w2, h2 = ax2.get_position().bounds
        ax1.set_position([0.1, 1.0-1.6*h2, w2, h2])
        ax2.set_position([0.1+w2+0.02, 1.0-1.6*h2, w2, h2])
        
        # --- Colorbar
        cbax = fig.add_axes([0.1, 1.0-0.5*h2, w1+w2+0.02, 0.03])
        cbar = plt.colorbar(c2, cax=cbax, orientation='horizontal')
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        if 'long_name' in roms_ds[variable].attrs:
            cbar.set_label(f'{roms_ds[variable].long_name} ({roms_ds[variable].units})')
        else:
            cbar.set_label(variable)

        # --- Set lon- and lat-range
        if lon_range is not None:
            ax1.set_xlim(lon_range)
            ax2.set_xlim(lon_range)
        if lat_range is not None:
            ax1.set_ylim(lat_range)
            ax2.set_ylim(lat_range)
        
        # --------------------------------------------------------
        # Transects
        # --------------------------------------------------------
        # --- Transects in maps
        ax1, leg1 = plot_transects(ax1, transects_to_plot, df_dswt, time_strs[t])
        leg1.remove()
        ax2, leg2 = plot_transects(ax2, transects_to_plot, df_dswt, time_strs[t])
        
        # --- Interactive transects
        lines = ax1.get_lines()
        for line in lines:
            line.set_picker(5)
        
        ax3 = plt.subplot(3, 2, (3, 4))
        ax4 = plt.subplot(3, 2, (5, 6), sharey=ax3)
        
        def on_pick(event): # !!! FIX !!! possible to keep transect selected while cycling through time?
            line = event.artist
            transect = next(item for item in transects_to_plot if item['name']==line.get_label())
            
            # --- Water column plot
            ax3.clear()
            transect_ds = select_roms_transect(roms_ds, transect['lon_land'], transect['lat_land'], transect['lon_ocean'], transect['lat_ocean'], 500.)
            transect_ds[variable][t, :, :].plot(x='distance', y='z_rho', vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=False, ax=ax3)
            
            l_dswt, _, _, _, _ = determine_dswt_along_transect(transect_ds, config)
            
            ax3.set_xlabel('Distance (m)')
            ax3.set_ylabel('Depth (m)')
            ax3.set_xlim([0, np.nanmax(transect_ds.distance.values)])
            ax3.set_ylim([np.nanmin(transect_ds.z_rho.values), 0])
            ax3.set_title(f'{transect["name"]} - DSWT: {l_dswt[t].astype(bool)}')
            
            # --- Vertical density gradient per cell along transect
            ax4.clear()
            for i in range(len(transect_ds.distance)):
                transect_ds.vertical_density_gradient[t, :, i].plot(y='z_rho')
            ax4.set_xlim([-0.005, 0.1])
            ax4.set_ylabel('Depth (m)')
            ax4.set_xlabel('Vertical density gradient (kg/m3/m)')
            ax4.set_title('')
            
            fig.canvas.draw()

        fig.canvas.mpl_connect('pick_event', on_pick)
        
        # --- Reposition axes
        l3, b3, w3, h3 = ax3.get_position().bounds
        ax3.set_position([0.1, 1.0-1.6*h2-h3-0.04, 2*w2+0.02, h3])
        ax4.set_position([0.1, 1.0-1.6*h2-2*h3-0.08, 2*w2+0.02, h3])
        
        # --- Overall title (time)
        title = f'{time_strs[t].strftime("%d-%m-%Y %H:%M")}'
        plt.suptitle(title)

    t = np.arange(0, len(roms_ds.ocean_time), t_interval)
    time = roms_ds.ocean_time.values[t]

    fig = plot_cycler(single_plot, time)
    plt.show()
    
if __name__ == '__main__':
    input_path = f'{get_dir_from_json("cwa")}2017/cwa_20170514_03__his.nc'
    grid_file = f'{get_dir_from_json("cwa")}grid.nc'
    roms_ds = load_roms_data(input_path, grid_file)
    
    lon_range = [114., 116.]
    lat_range = [-33.0, -31.0]
    transects = get_transects_in_lon_lat_range('input/transects/cwa_transects.json', lon_range, lat_range)
    
    config = read_config('cwa')
    
    df_dswt = pd.read_csv('output/cwa_114-116E_33-31S/old_dswt_2017.csv', index_col=['time', 'transect']) # (this is a MultiIndex DataFrame)
    
    plot_dswt_maps_transects(roms_ds, transects, df_dswt, config)
