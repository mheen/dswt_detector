import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.io import shapereader
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

lon_range_default = [110.0, 154.0]
lat_range_default = [-46.0, -24.0]

def plot_contours(lon:np.ndarray, lat:np.ndarray, h:np.ndarray,
                  lon_range=None, lat_range=None, ax=None, show=False,
                  fontsize=10, color='k', linewidths=1,
                  clevels=[50, 100, 200, 400, 800, 1000],
                  clabel=True) -> plt.axes:

    if lon_range is None:
        lon_range = lon_range_default
    if lat_range is None:
        lat_range = lat_range_default

    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, lon_range=lon_range, lat_range=lat_range)
        
    def _fmt(x):
        s = f'{x:.0f}'
        return s

    # get masked arrays so contour labels fall within plot if range is limited
    mask = np.ones(h.shape).astype(bool)
    mask = mask & (lon>=lon_range[0]) & (lon<=lon_range[1])
    mask = mask & (lat>=lat_range[0]) & (lat<=lat_range[1])
    xm = np.ma.masked_where(~mask, lon)
    ym = np.ma.masked_where(~mask, lat)
    zm = np.ma.masked_where(~mask, h)

    cs = ax.contour(xm, ym, zm, levels=clevels,
                    colors=color, linewidths=linewidths, transform=ccrs.PlateCarree())
    if clabel is True:
        ax.clabel(cs, cs.levels, fontsize=fontsize, inline=True, fmt=_fmt)

    if show is True:
        plt.show()
    else:
        return ax

def plot_bathymetry(lon:np.ndarray, lat:np.ndarray, h:np.ndarray,
                    lon_range=None, lat_range=None, ax=None, show=False,
                    cmap='BrBG', vmin=None, vmax=None) -> plt.axes:

    if lon_range is None:
        lon_range = lon_range_default
    if lat_range is None:
        lat_range = lat_range_default

    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, lon_range=lon_range, lat_range=lat_range)
    
    c = ax.pcolormesh(lon, lat, h, cmap=cmap, vmin=vmin, vmax=vmax)

    l,b,w,h = ax.get_position().bounds
    fig = plt.gcf()
    cbax = fig.add_axes([l+w+0.02, b, 0.05*w, h])
    cbar = plt.colorbar(c, cax=cbax)
    cbar.set_label('Bathymetry (m)')

    if show is True:
        plt.show()
    else:
        return ax, c, cbar

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
                   draw_grid=False, full_resolution=False) -> plt.axes:
    
    if lon_range is None:
        lon_range = lon_range_default
    if lat_range is None:
        lat_range = lat_range_default
    if meridians is None:
        meridians = np.arange(110.0, 160.0, 10.0)
    if parallels is None:
        parallels = np.arange(-46.0, -24.0, 4.0)
        
    if full_resolution == True:
        coast = cfeature.GSHHSFeature(scale="full")
        ax.add_feature(coast, linewidth=1, edgecolor='k', facecolor='#d2d2d2', zorder=2)
    else:
        ax.add_feature(cfeature.LAND, edgecolor='k', facecolor='#d2d2d2')
        ax.add_feature(cfeature.COASTLINE)
    
    ax = add_grid(ax, meridians, parallels, xmarkers, ymarkers, draw_grid)

    ax.set_extent([lon_range[0], lon_range[1],
                   lat_range[0], lat_range[1]],
                   ccrs.PlateCarree())
    
    return ax
