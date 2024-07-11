import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.config import Config, read_config
from transects import read_transects_dict_from_json
from transects import get_depth_contours
from tools.files import get_dir_from_json
from tools import log

from rasterio import features
import shapely
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import numpy as np
import xarray as xr
import json

def convert_land_mask_to_polygons(lon:np.ndarray[float], lat:np.ndarray[float], mask:np.ndarray[int]) -> list[shapely.Polygon]:
    land_mask = np.abs(mask-1).astype(np.int32) # assuming ocean points are 1 in mask!
    
    shapes = features.shapes(land_mask, mask=land_mask.astype(bool)) # mask so that only land polygons are returned
    
    # add extra column and row to lon and lat because polygon goes around edges
    # note: this is obviously not the best way to do this, but errors seem to be minor
    lon_extended = np.hstack((lon, np.expand_dims(lon[:, -1]+np.diff(lon, axis=1)[:, -1], 1)))
    lon_extended = np.vstack((lon_extended, lon_extended[-1, :]))
    lat_extended = np.hstack((lat, np.expand_dims(lat[:, -1], 1)))
    lat_extended = np.vstack((lat_extended, lat_extended[-1, :]+np.diff(lat_extended, axis=0)[-1, :]))
    
    land_polys = []
    for vec, _ in shapes:
        i, j = shapely.geometry.shape(vec).exterior.xy
        i = np.array(i).astype(int)
        j = np.array(j).astype(int)
        land_polys.append(shapely.Polygon(list(zip(lon_extended[j, i], lat_extended[j, i]))))
        
    return land_polys

def map_with_land_and_contours(ax:plt.axes,
                               contours:list[shapely.LineString],
                               land_polygons:list[shapely.Polygon],
                               lon_range:list, lat_range:list,
                               meridians:list, parallels:list):
    
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.set_xticks(meridians, crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_yticks(parallels, crs=ccrs.PlateCarree())

    ax.set_extent([lon_range[0], lon_range[1],
                   lat_range[0], lat_range[1]],
                   ccrs.PlateCarree())
    
    # plot land mask
    for land_polygon in land_polygons:
        x, y = land_polygon.exterior.xy
        ax.fill(x, y, edgecolor='k', facecolor='#d2d2d2')
        
    # plot contours
    for i in range(len(contours)):
        x, y = contours[i].coords.xy
        ax.plot(x, y, '-k', linewidth=0.5, label=f'c{i}')
        
    return ax

def remove_transects_in_plot(transects_file:str,
                             grid_ds:xr.Dataset,
                             config:Config,
                             lon_range=None, lat_range=None,
                             transect_interval=1,
                             color='#C70039',
                             linewidth=1.0,
                             vmin=1,
                             vmax=5,
                             cmap='cividis',
                             meridians=None,
                             parallels=None) -> plt.axes:
    
    transects = read_transects_dict_from_json(transects_file)
    contours = get_depth_contours(grid_ds.lon_rho.values, grid_ds.lat_rho.values, grid_ds.h.values, config.transect_contours)
    land_polygons = convert_land_mask_to_polygons(grid_ds.lon_rho.values, grid_ds.lat_rho.values, grid_ds.mask_rho.values)
    
    if lon_range == None:
        lon_range = [np.floor(np.nanmin(grid_ds.lon_rho.values)), np.ceil(np.nanmax(grid_ds.lon_rho.values))]
    if lat_range == None:
        lat_range = [np.floor(np.nanmin(grid_ds.lat_rho.values)), np.ceil(np.nanmax(grid_ds.lat_rho.values))]
    
    if meridians == None:
        dx = (lon_range[1]-lon_range[0])/3
        meridians = np.arange(lon_range[0], lon_range[1]+dx, dx)
    if parallels == None:
        dy = (lat_range[1]-lat_range[0])/4
        parallels = np.arange(lat_range[0], lat_range[1]+dy, dy)
    
    fig = plt.figure(figsize=(12, 8))
    plt.suptitle('Click on the transects you want to remove, then close the figure', y=0.93)
    
    # (a) transects
    ax = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax.set_title('Transects')
    map_with_land_and_contours(ax, contours, land_polygons, lon_range, lat_range, meridians, parallels)
    
    # plot transects
    transect_names = list(transects.keys())
    
    for i in np.arange(0, len(transect_names), transect_interval):
        lon_org = transects[transect_names[i]]['lon_org']
        lat_org = transects[transect_names[i]]['lat_org']
        
        ax.plot(lon_org, lat_org, '-', color=color, linewidth=linewidth, label=transect_names[i])
        ax.text(lon_org[0], lat_org[0], transect_names[i], ha='right', va='center', fontsize=8)
        
    # (b) grid cell coverage
    grid_coverage = np.zeros(grid_ds.lon_rho.shape)
    
    for i in range(len(transect_names)):
        eta = transects[transect_names[i]]['eta']
        xi = transects[transect_names[i]]['xi']
        grid_coverage[eta, xi] += 1
    
    grid_coverage[grid_coverage==0] = np.nan
    
    ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax2.set_title('Grid cell coverage')
    map_with_land_and_contours(ax2, contours, land_polygons, lon_range, lat_range, meridians, parallels)
    ax2.set_yticklabels([])
    c = ax2.pcolormesh(grid_ds.lon_rho.values, grid_ds.lat_rho.values, grid_coverage, vmin=vmin, vmax=vmax, cmap=cmap)
    # colorbar
    l2, b2, w2, h2 = ax2.get_position().bounds
    cax = fig.add_axes([l2+w2+0.02, b2, 0.02, h2])
    cbar = plt.colorbar(c, cax=cax)
    cbar.set_label('Grid cell in transects (#)')
    cbar.set_ticks(np.arange(vmin, vmax+1, 1))
    
    # --- Interactive transect removal
    lines = ax.get_lines() # this also retrieves contour lines..
    for line in lines:
        line.set_picker(1)
    
    removed_transects = []
    
    def on_pick(event):
        line = event.artist
        remove_transect = line.get_label()
        
        if remove_transect.startswith('t'):
            line.remove()
            
            eta = transects[remove_transect]['eta']
            xi = transects[remove_transect]['xi']
            grid_coverage[eta, xi] -= 1
            grid_coverage[grid_coverage==0] = np.nan

            # FIX: figure not updating properly
            ax2.update_from(ax2.pcolormesh(grid_ds.lon_rho.values, grid_ds.lat_rho.values,
                                           grid_coverage, vmin=vmin, vmax=vmax, cmap=cmap))
            
            removed_transects.append(remove_transect)

            plt.draw()
    
    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()
    
    return removed_transects

def remove_transects_from_file(remove_transects:list, output_path:str):
    confirm = input(f'''Transects selected to remove: {remove_transects}
          Do you want to save to file? Y/N''')
    if confirm.lower().startswith('y'):
        transects = read_transects_dict_from_json(output_path)
        for t in remove_transects:
            transects.pop(t)
    
        log.info(f'Writing transects to json file: {output_path}')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transects, f, ensure_ascii=False, indent=4)
            
    else:
        log.info('Did not make any changes to transects file.')

def interactive_transect_removal(transects_file:str, grid_ds:xr.Dataset, config:Config,
                                 lon_range=None, lat_range=None, transect_interval=1,
                                 color='#C70039', linewidth=1.0, vmin=1, vmax=5, cmap='cividis',
                                 meridians=None, parallels=None) -> bool:
    removed_transects_bool = False
    remove_transects = remove_transects_in_plot(transects_file, grid_ds, config,
                                                lon_range=lon_range, lat_range=lat_range,
                                                transect_interval=transect_interval, color=color,
                                                linewidth=linewidth, vmin=vmin, vmax=vmax,
                                                cmap=cmap, meridians=meridians, parallels=parallels)
    if len(remove_transects) != 0:
        removed_transects_bool = True
        remove_transects_from_file(remove_transects, transects_file)
    else:
        log.info('No transects selected to remove, did not make any changes to transects file.')
        
    return removed_transects_bool

if __name__ == '__main__':
    transects_file = 'input/transects/cwa_transects.json'
    
    grid_file = f'{get_dir_from_json("cwa")}grid.nc'
    grid_ds = xr.load_dataset(grid_file)
    
    config = read_config('cwa')
    
    interactive_transect_removal(transects_file, grid_ds, config)
