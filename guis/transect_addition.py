import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from guis.transect_removal import map_with_land_and_contours, convert_land_mask_to_polygons
from tools.config import Config, read_config
from transects import read_transects_in_lon_lat_range_from_json, read_transects_dict_from_json
from transects import get_depth_contours, get_starting_points, find_transects_from_starting_points
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

def transect_addition_plot(transects_file:str,
                           grid_ds:xr.Dataset,
                           config:Config,
                           lon_range=None, lat_range=None,
                           transect_interval=1,
                           color='#C70039',
                           linewidth=1.0,
                           meridians=None,
                           parallels=None) -> tuple[plt.figure, plt.axes]:
        
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
    
    proj = ccrs.PlateCarree()
    
    fig = plt.figure(figsize=(12, 8))
    plt.suptitle('Press "n" to start selecting an area to add transects.')
    
    ax = plt.axes(projection=proj)
    
    map_with_land_and_contours(ax, contours, land_polygons, lon_range, lat_range, meridians, parallels)
    
    # plot transects
    transect_names = list(transects.keys())
    
    for i in np.arange(0, len(transect_names), transect_interval):
        lon_org = transects[transect_names[i]]['lon_org']
        lat_org = transects[transect_names[i]]['lat_org']
        
        ax.plot(lon_org, lat_org, '-', color=color, linewidth=linewidth)
        ax.text(lon_org[0], lat_org[0], transect_names[i], ha='right', va='center', fontsize=8)
        
    return fig, ax

class InteractiveTransectAddition:
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.lons = []
        self.lats = []
        self.c = None
        
        self.is_click_enabled = False
        self.cid_click = None
        
        lines = ax.get_lines() # make contour lines selectable
        for line in lines:
            line.set_picker(4)
            
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
    
    def _plot_area(self):
        if len(self.lons) > 2:
            self.ax.lines[-1].remove()
            poly = shapely.Polygon([(self.lons[i], self.lats[i]) for i in range(len(self.lons))])
            
            x, y = poly.exterior.xy
            self.ax.scatter(self.lons, self.lats, c='b')
            self.ax.plot(x, y, '-b')
            
        elif len(self.lons) == 2:
            
            line = shapely.LineString([(self.lons[i], self.lats[i]) for i in range(len(self.lons))])
            
            x, y = line.coords.xy
            self.ax.scatter(self.lons, self.lats, c='b')
            self.ax.plot(x, y, '-b')
        
        else:
            self.ax.scatter(self.lons[0], self.lats[0], c='b')
            
        self.fig.canvas.draw()

    def on_press(self, event):
        if event.key == 'n':
            self.is_click_enabled = True
            plt.suptitle('''Click to create a polygon in which to add transects.
                         Press "x" to remove the last point.
                         Press "y" when you are done.''')
            self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            self.fig.canvas.draw()
            
        if event.key == 'x':
            self.ax.scatter(self.lons[-1], self.lats[-1], c='k')
            self.lons.remove(self.lons[-1])
            self.lats.remove(self.lats[-1])
            
            self._plot_area()
            
        if event.key == 'y':
            self.is_click_enabled = False
            self.fig.canvas.mpl_disconnect(self.cid_click)
            plt.suptitle('''Click on the contourline from which to want transects to start.
                         Close the figure when you are done.''')
            # fig.canvas.mpl_connect('button_press_event', stop_click)
            self.fig.canvas.mpl_connect('pick_event', self.on_pick)
            self.fig.canvas.draw()

    def on_click(self, event):
        proj = ccrs.PlateCarree()
        lon, lat = proj.transform_point(event.xdata, event.ydata, proj)
        self.lons.append(lon)
        self.lats.append(lat)
        
        self._plot_area()
        
    def on_pick(self, event):
        line = event.artist
        contour_label = line.get_label()
        
        if contour_label.startswith('c'):
            contour_i = int(contour_label.replace('c', ''))
            print(f'Contour: {contour_i}')
            self.c = contour_i
            
            plt.suptitle(f'''Selected contour {contour_i} to start transects from.
                         Click on another contour if this is not what you wanted.
                         Close the figure when you are done.''')
            self.fig.canvas.draw()
    
    def show(self):
        plt.show()

def add_transects(add_polygon:shapely.Polygon, i_start_contour:int,
                  ds_grid:xr.Dataset, config:Config, output_path:str) -> bool:
    
    added_transects_bool = False
    # get approximate grid resolution (in degrees):
    dx = np.nanmean(np.unique(np.diff(ds_grid.lon_rho.values, axis=1)))
    dy = np.nanmean(np.unique(np.diff(ds_grid.lat_rho.values, axis=0)))
    ds = np.nanmin([dx, dy])

    # get depth contours
    contours = get_depth_contours(ds_grid.lon_rho.values,
                                  ds_grid.lat_rho.values,
                                  ds_grid.h.values,
                                  config.transect_contours[i_start_contour:])
    # get starting points along shallowest depth contour
    lon_ps, lat_ps = get_starting_points(contours[0], ds)
    
    # get only starting points within polygon
    l_polygon = [add_polygon.contains(shapely.Point(lon_ps[i], lat_ps[i])) for i in range(len(lon_ps))]
    lon_ps = lon_ps[l_polygon]
    lat_ps = lat_ps[l_polygon]
    
    transects_all = read_transects_dict_from_json(output_path)
    transect_keys = np.sort(np.array([int(t.replace('t', '')) for t in transects_all.keys()]))
    new_transects = find_transects_from_starting_points(ds_grid, contours, lon_ps, lat_ps, ds, config,
                                                    start_index=transect_keys[-1]+1)
    
    if len(new_transects) != 0:
        added_transects_bool = True
        transects_all.update(new_transects) # add new transects to existing ones

        log.info(f'Appending additional transects to json file: {output_path}')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transects_all, f, ensure_ascii=False, indent=4)
        
    return added_transects_bool

def interactive_transect_addition(transects_file:str, grid_ds:xr.Dataset, config:Config,
                                  lon_range=None, lat_range=None, transect_interval=1,
                                  color='#C70039', linewidth=1.0, meridians=None, parallels=None) -> bool:
    
    fig, ax = transect_addition_plot(transects_file, grid_ds, config,
                                     lon_range=lon_range, lat_range=lat_range, transect_interval=transect_interval,
                                     color=color, linewidth=linewidth, meridians=meridians, parallels=parallels)
    interactive_addition = InteractiveTransectAddition(fig, ax)
    interactive_addition.show()
    add_polygon = shapely.Polygon(list(zip(interactive_addition.lons, interactive_addition.lats)))

    added_transects_bool = add_transects(add_polygon, interactive_addition.c, grid_ds, config, transects_file)
    
    return added_transects_bool

if __name__ == '__main__':
    transects_file = 'input/transects/cwa_transects.json'
    
    grid_file = f'{get_dir_from_json("cwa")}grid.nc'
    grid_ds = xr.load_dataset(grid_file)
    
    config = read_config('cwa')
    
    interactive_transect_addition(transects_file, grid_ds, config)
