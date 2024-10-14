import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from plot_tools.basic_maps import plot_basic_map
from plot_tools.general import add_subtitle
from readers.read_ocean_data import select_roms_transect_from_known_coordinates
from dswt.dswt_detection import determine_dswt_along_transect

from tools.config import read_config, Config
from readers.read_ocean_data import load_roms_data, select_roms_subset
from tools.files import get_dir_from_json
from transects import read_transects_in_lon_lat_range_from_json

import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cmocean as cm
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
from datetime import datetime

lon_range_default = [114.5, 116.]
lat_range_default = [-33.0, -31.0]

def select_transects_to_plot(transects:dict, transect_interval:int) -> list[dict]:
    
    transect_names = list(transects.keys())
    
    transects_to_plot = []
    for i in np.arange(0, len(transect_names), transect_interval):
        lon_org = transects[transect_names[i]]['lon_org']
        lat_org = transects[transect_names[i]]['lat_org']
        eta = transects[transect_names[i]]['eta']
        xi = transects[transect_names[i]]['xi']
        
        transect = {'name': transect_names[i],
                    'lon_org': lon_org, 'lat_org': lat_org,
                    'eta': eta, 'xi': xi}

        transects_to_plot.append(transect)
        
    return transects_to_plot

def get_dswt_transect(df_dswt:pd.DataFrame, transect_name:str, time:datetime) -> bool:
    f_dswt = df_dswt.xs(time.strftime("%Y-%m-%d"), level='time', drop_level=False).xs(transect_name, level='transect', drop_level=False)['f_dswt']
    if f_dswt.values[0] > 0.0:
        return True
    else:
        return False

def plot_transects(ax:plt.axes, transects:list[dict], df_dswt:pd.DataFrame, time:datetime,
                   linewidth=0.7, loc='upper right') -> tuple[plt.axes, plt.legend]:
    for transect in transects:
        if df_dswt is not None:
            l_dswt = get_dswt_transect(df_dswt, transect, time)
        else:
            l_dswt = None
        if l_dswt == True:
            ax.plot(transects[transect]['lon_org'], transects[transect]['lat_org'], '-', linewidth=linewidth, color='#1b7931', label=transect)
        elif l_dswt == False:
            ax.plot(transects[transect]['lon_org'], transects[transect]['lat_org'], '-', linewidth=linewidth, color='k', label=transect)
        else:
            ax.plot(transects[transect]['lon_org'], transects[transect]['lat_org'], '-', linewidth=linewidth, color='#989898', label=transect)
    # legend
    legend_elements = [Line2D([0], [0], color='#1b7931'), Line2D([0], [0], color='k'), Line2D([0], [0], color='#989898')]
    l = ax.legend(legend_elements, ['DSWT', 'no DSWT', 'unknown'], loc=loc, bbox_to_anchor=(1.01, 1.0))
    
    return ax, l

def base_plot(fig):
    
    ax1 = plt.subplot(4, 4, (1, 5), projection=ccrs.PlateCarree())
    ax2 = plt.subplot(4, 4, (2, 6), projection=ccrs.PlateCarree())
    ax3 = plt.subplot(4, 4, (3, 8))
    ax4 = plt.subplot(4, 4, (9, 10))
    ax5 = plt.subplot(4, 4, (11, 12))
    ax6 = plt.subplot(4, 4, (13, 14))
    ax7 = plt.subplot(4, 4, (15, 16))
    
    l3, b3, w3, h3 = ax3.get_position().bounds
    cax1 = fig.add_axes([l3+w3+0.02, b3, 0.02, h3])
    
    l4, b4, w4, h4 = ax4.get_position().bounds
    cax4 = fig.add_axes([l4-0.04, b4, 0.02, h4])
    
    l5, b5, w5, h5 = ax5.get_position().bounds
    cax5 = fig.add_axes([l5+w5+0.02, b5, 0.02, h5])
    
    l6, b6, w6, h6 = ax6.get_position().bounds
    cax6 = fig.add_axes([l6-0.04, b6, 0.02, h6])
    
    # hide all axes except for maps
    ax3.set_axis_off()
    ax4.set_axis_off()
    ax5.set_axis_off()
    ax6.set_axis_off()
    ax7.set_axis_off()
    
    cax1.set_axis_off()
    cax4.set_axis_off()
    cax5.set_axis_off()
    cax6.set_axis_off()
    
    return ax1, ax2, ax3, ax4, ax5, ax6, ax7, cax1, cax4, cax5, cax6

class InteractiveTransectSelectionTimeStepping:
    def __init__(self,
                 roms_ds:xr.Dataset,
                 transects_to_plot:dict,
                 df_dswt:pd.DataFrame,
                 config:Config,
                 lon_range,
                 lat_range):
        
        self.roms_ds = roms_ds
        self.roms_ds_ss = select_roms_subset(roms_ds, None, lon_range, lat_range)
        self.transects_to_plot = transects_to_plot
        self.time = roms_ds['ocean_time'].values
        
        self.transect = None
        self.transect_ds = None
        
        self.df_dswt = df_dswt
        
        self.config = config
        
        self.fig = plt.figure(figsize=(9, 8))
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6, self.ax7, self.cax1, self.cax4, self.cax5, self.cax6 = base_plot(self.fig)
        
        self.vmin = {'rho': 1025.0, 'temp':18.0, 'salt':35.0, 'drhodz': 0.0}
        self.vmax = {'rho': 1026.0, 'temp':22.0, 'salt':36.0, 'drhodz': 0.03}
        self.cmap = {'rho': cm.cm.thermal_r, 'temp': 'RdYlBu_r', 'salt':cm.cm.haline, 'drhodz': 'bone_r'}
        
        # time index to plot
        self.t = 0
        self._update_vmin_vmax()
        
        self._map_plots()
        
        self.picked_transect = False
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
    
    def _update_vmin_vmax(self):
        
        def _get_min_max(values, buffer_min=0.05, buffer_max=0.2):
            min_value = np.nanmin(values)
            max_value = np.nanmax(values)
            dvalue = max_value-min_value
            return min_value+buffer_min*dvalue, max_value-buffer_max*dvalue
        
        self.vmin['rho'], self.vmax['rho'] = _get_min_max(self.roms_ds_ss['density'][self.t, :, :, :].values, buffer_max=0.8, buffer_min=0)
        self.vmin['temp'], self.vmax['temp'] = _get_min_max(self.roms_ds_ss['temp'][self.t, :, :, :].values, buffer_max=0.05, buffer_min=0.8)
        self.vmin['salt'], self.vmax['salt'] = _get_min_max(self.roms_ds_ss['salt'][self.t, :, :, :].values, buffer_max=0.05)
    
    def _map_plots(self):
        self.ax1.clear()
        self.ax2.clear()
        self.cax1.clear()
        
        plt.suptitle(f'''{pd.to_datetime(self.time[self.t]).strftime("%d %b %Y %H:%M")}
                     Click on a transect in the left panel to plot parameters along the transect.
                     Use left/right arrows to cycle through in time.''', y=0.99)
        
        # density maps
        plot_basic_map(self.ax1, self.lon_range, self.lat_range)
        _, leg1 = plot_transects(self.ax1, self.transects_to_plot, self.df_dswt, pd.to_datetime(self.time[self.t]))
               
        c1 = self.roms_ds['density'][self.t, -1, :, :].plot(ax=self.ax1, transform=ccrs.PlateCarree(), x='lon_rho', y='lat_rho',
                                                            vmin=self.vmin['rho'], vmax=self.vmax['rho'],
                                                            cmap=self.cmap['rho'], add_colorbar=False)
        self.ax1.set_extent([self.lon_range[0], self.lon_range[1],
                             self.lat_range[0], self.lat_range[1]], ccrs.PlateCarree())
        self.ax1.set_ylabel('')
        self.ax1.set_xlabel('')
        self.ax1.set_title('Surface')
        
        plot_basic_map(self.ax2, self.lon_range, self.lat_range)
        _, leg2 = plot_transects(self.ax2, self.transects_to_plot, self.df_dswt, pd.to_datetime(self.time[self.t]))
        c2 = self.roms_ds['density'][self.t, 0, :, :].plot(ax=self.ax2, transform=ccrs.PlateCarree(), x='lon_rho', y='lat_rho',
                                                            vmin=self.vmin['rho'], vmax=self.vmax['rho'],
                                                            cmap=self.cmap['rho'], add_colorbar=False)
        self.ax2.set_extent([self.lon_range[0], self.lon_range[1],
                             self.lat_range[0], self.lat_range[1]], ccrs.PlateCarree())
        self.ax2.set_ylabel('')
        self.ax2.set_xlabel('')
        self.ax2.set_title('Bottom')
        
        # resize maps
        l1, b1, w1, h1 = self.ax1.get_position().bounds
        l2, b2, w2, h2 = self.ax2.get_position().bounds
        l3, b3, w3, h3 = self.ax3.get_position().bounds
        
        self.ax1.set_position([l1, b3, w1/h1*h3, h3])
        self.ax2.set_position([l2, b3, w2/h2*h3, h3])
        
        # legend
        leg1.remove()
        leg1.set_bbox_to_anchor((-0.5, 1.0))
        
        # making transects selectable
        lines = self.ax1.get_lines()
        for line in lines:
            line.set_picker(5)
        
        cbar = plt.colorbar(c1, cax=self.cax1)
        
        plt.draw()
    
    def _transect_plots(self):
        self.ax3.clear()
        self.ax4.clear()
        self.ax5.clear()
        self.ax6.clear()
        self.ax7.clear()
        self.cax4.clear()
        self.cax5.clear()
        self.cax6.clear()
        
        if self.picked_transect == True:
            transect = self.transects_to_plot[self.transect]
            self.transect_ds = select_roms_transect_from_known_coordinates(self.roms_ds, transect['eta'], transect['xi'])
            
            # density
            self.transect_ds['density'][self.t, :, :].plot(x='distance', y='z_rho', vmin=self.vmin['rho'], vmax=self.vmax['rho'],
                                                        cmap=self.cmap['rho'], add_colorbar=False, ax=self.ax3)
            l_dswt, _, _, _, _ = determine_dswt_along_transect(self.transect_ds, self.config)
            self.ax3.set_title(f'{self.transect} - DSWT: {l_dswt[self.t].astype(bool)}')
            
            self.ax3.fill_between(self.transect_ds.distance.values, -210, -self.transect_ds.h.values, color='#d2d2d2', edgecolor='k')
            self.ax3.set_xlim([self.transect_ds.distance.values[0], self.transect_ds.distance.values[-1]])
            self.ax3.set_ylim([-200, 0])
            self.ax3.set_xticklabels([])
            self.ax3.set_xlabel('')
            self.ax3.set_ylabel('Depth')
            self.ax3.yaxis.set_label_coords(0.07, 0.5, transform=self.ax3.transAxes)
            self.ax3.set_yticks([0, -50, -100, -150, -200])
            self.ax3.set_yticklabels([])
            add_subtitle(self.ax3, 'Density (kg/m$^3$)', location='lower left')
            
            # temperature
            c4 = self.transect_ds['temp'][self.t, :, :].plot(x='distance', y='z_rho', vmin=self.vmin['temp'], vmax=self.vmax['temp'],
                                                        cmap=self.cmap['temp'], add_colorbar=False, ax=self.ax4)
            self.ax4.fill_between(self.transect_ds.distance.values, -210, -self.transect_ds.h.values, color='#d2d2d2', edgecolor='k')
            self.ax4.set_xlim([self.transect_ds.distance.values[0], self.transect_ds.distance.values[-1]])
            self.ax4.set_ylim([-200, 0])
            self.ax4.set_xticklabels([])
            self.ax4.set_xlabel('')
            self.ax4.set_ylabel('Depth')
            self.ax4.yaxis.set_label_coords(0.07, 0.5, transform=self.ax4.transAxes)
            self.ax4.set_yticks([0, -50, -100, -150, -200])
            self.ax4.set_yticklabels([])
            self.ax4.set_title('')
            add_subtitle(self.ax4, 'Temperature ($^o$C)', location='lower left')
            
            # salinity
            c5 = self.transect_ds['salt'][self.t, :, :].plot(x='distance', y='z_rho', vmin=self.vmin['salt'], vmax=self.vmax['salt'],
                                                        cmap=self.cmap['salt'], add_colorbar=False, ax=self.ax5)
            self.ax5.fill_between(self.transect_ds.distance.values, -210, -self.transect_ds.h.values, color='#d2d2d2', edgecolor='k')
            self.ax5.set_xlim([self.transect_ds.distance.values[0], self.transect_ds.distance.values[-1]])
            self.ax5.set_ylim([-200, 0])
            self.ax5.set_ylabel('Depth')
            self.ax5.yaxis.set_label_coords(0.07, 0.5, transform=self.ax5.transAxes)
            self.ax5.set_yticks([0, -50, -100, -150, -200])
            self.ax5.set_yticklabels([])
            self.ax5.set_xticklabels([])
            self.ax5.set_xlabel('')
            self.ax5.set_title('')
            add_subtitle(self.ax5, 'Salinity', location='lower left')
            
            # vertical density gradient
            c6 = self.transect_ds['vertical_density_gradient'][self.t, :, :].plot(x='distance', y='z_rho', vmin=self.vmin['drhodz'], vmax=self.vmax['drhodz'],
                                                        cmap=self.cmap['drhodz'], add_colorbar=False, ax=self.ax6)
            self.ax6.fill_between(self.transect_ds.distance.values, -210, -self.transect_ds.h.values, color='#d2d2d2', edgecolor='k')
            self.ax6.set_xlim([self.transect_ds.distance.values[0], self.transect_ds.distance.values[-1]])
            self.ax6.set_ylim([-200, 0])
            self.ax6.set_xlabel('')
            self.ax4.set_xticklabels([])
            self.ax6.set_ylabel('Depth')
            self.ax6.yaxis.set_label_coords(0.07, 0.5, transform=self.ax6.transAxes)
            self.ax6.set_yticks([0, -50, -100, -150, -200])
            self.ax6.set_yticklabels([])
            self.ax6.set_title('')
            add_subtitle(self.ax6, 'Vertical density gradient (kg/m$^3$/m)', location='lower left')
            
            # vertical density gradient in each cell
            for i in range(len(self.transect_ds.distance)):
                self.transect_ds.vertical_density_gradient[self.t, :, i].plot(y='z_rho', ax=self.ax7)
            self.ax7.set_xlim([self.vmin['drhodz'], self.vmax['drhodz']+0.5*self.vmax['drhodz']])
            self.ax7.set_ylim([-200, 0])
            self.ax7.set_ylabel('Depth (m)')
            self.ax7.set_yticks([0, -50, -100, -150, -200])
            self.ax7.set_yticklabels([0, 50, 100, 150, 200])
            self.ax7.yaxis.tick_right()
            self.ax7.yaxis.set_label_position('right')
            self.ax7.set_xlabel('Vertical density gradient (kg/m$^3$/m)')
            self.ax7.set_title('')
            
            # colorbars
            cbar4 = plt.colorbar(c4, cax=self.cax4)
            self.cax4.yaxis.set_ticks_position('left')
            
            cbar5 = plt.colorbar(c5, cax=self.cax5)
            
            cbar6 = plt.colorbar(c6, cax=self.cax6)
            self.cax6.yaxis.set_ticks_position('left')
            
            plt.draw()
        
    def update_plot(self):
        self._update_vmin_vmax()
        self._map_plots()
        
        if self.picked_transect == True:
            # update transects
            self._transect_plots()
    
    def on_pick(self, event):
        self.picked_transect = True
        
        line = event.artist
        self.transect = next(item for item in self.transects_to_plot if item==line.get_label())
        
        self._transect_plots()

    def on_press(self, event):
        if (event.key == "left"):
            if self.t <= 0:
                self.t = len(self.time) -1
            self.t -= 1
            self.update_plot()
        if (event.key == "right"):
            if self.t >= len(self.time) - 1:
                self.t = 0
            self.t += 1
            self.update_plot()
        if (event.key == "up"):
            self.t = len(self.time) - 1
            self.update_plot()
        if (event.key == "down"):
            self.t = 0
            self.update_plot()
            
    def show(self):
        plt.show()
    
if __name__ == '__main__':
    input_path = f'{get_dir_from_json("cwa")}2017/cwa_20170211_03__his.nc'
    grid_file = f'{get_dir_from_json("cwa")}grid.nc'
    roms_ds = load_roms_data(input_path, grid_file)
    
    lon_range = [115., 116.]
    lat_range = [-33.0, -31.0]
    transects = read_transects_in_lon_lat_range_from_json('input/transects/cwa_transects.json', lon_range, lat_range)
    
    config = read_config('cwa')
    
    # df_dswt = pd.read_csv('output/cwa_114-116E_33-31S/dswt_2017.csv', index_col=['time', 'transect']) # (this is a MultiIndex DataFrame)
    df_dswt = None
    interactive_plot = InteractiveTransectSelectionTimeStepping(roms_ds, transects, df_dswt, config,
                                                                lon_range, lat_range)
    interactive_plot.show()
