import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.io import shapereader
import matplotlib.pyplot as plt
import numpy as np

def add_grid(ax:plt.axes, meridians:list, parallels:list,
              xmarkers:str, ymarkers:str, draw_grid:bool) -> plt.axes:

    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()

    ax.xaxis.set_major_formatter(lon_formatter)
    ax.set_xticks(meridians, crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_yticks(parallels, crs=ccrs.PlateCarree())

    if xmarkers == 'top':
        ax.xaxis.tick_top()
    if xmarkers == 'off':
        ax.set_yticklabels([])
    if ymarkers == 'right':
        ax.yaxis.tick_right()
    if ymarkers == 'off':
        ax.set_yticklabels([])

    if draw_grid is True:
        ax.grid(b=True, linewidth=0.5, color='k', linestyle=':', zorder=10)

    return ax

def plot_basic_map(ax:plt.axes, lon_range=None, lat_range=None,
                   meridians=None, parallels=None,
                   xmarkers='bottom', ymarkers='left',
                   draw_grid=False, zorder_c=5) -> plt.axes:
    
    if lon_range is None:
        lon_range = [110.0, 154.0]
    if lat_range is None:
        lat_range = [-46.0, -24.0]
    if meridians is None:
        meridians = np.arange(110.0, 160.0, 10.0)
    if parallels is None:
        parallels = np.arange(-46.0, -24.0, 4.0)
        
    shp = shapereader.Reader('input/GSHHS_coastline_GSR.shp')
    for _, geometry in zip(shp.records(), shp.geometries()):
        ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='#989898',
                           edgecolor='black', zorder=zorder_c)
        
    # shp2 = shapereader.Reader('input/GI.shp')
    # for _, geometry in zip(shp2.records(), shp2.geometries()):
    #     ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='#989898',
    #                       edgecolor='black', zorder=100)
    
    ax = add_grid(ax, meridians, parallels, xmarkers, ymarkers, draw_grid)

    ax.set_extent([lon_range[0], lon_range[1],
                   lat_range[0], lat_range[1]],
                   ccrs.PlateCarree())
    
    return ax
