from readers.read_dswt_output import get_transport_map

from tools.timeseries import get_monthly_means, get_monthly_climatology, get_yearly_means
from plot_tools.basic_timeseries import plot_histogram_multiple_years, plot_yearly_grid, plot_monthly_grid
from plot_tools.general import add_subtitle
from plot_tools.basic_maps import plot_basic_map, plot_contours

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from datetime import datetime
import pandas as pd
import xarray as xr

color_dswt = '#25419e'
color_transport = '#0e6e22'
color_pos = '#900C3F'
color_neg = '#1e1677'

lon_range_default = [114.5, 116.]
lat_range_default = [-33., -31.]
meridians_default = [115., 116.]
parallels_default = [-33., -32., -31.]

def plot_dswt_timeseries(df_timeseries:pd.DataFrame,
                         depth_contour:float,
                         years:list[int],
                         output_path=None,
                         show=True):
    
    time = np.array([pd.to_datetime(t) for t in df_timeseries['time'].values])
    f_dswt = df_timeseries['f_dswt'].values
    transport_contour = df_timeseries[f'transport_{str(int(depth_contour))}m']
    
    xlim = [time[0], time[-1]]
    
    time_m, f_dswt_m, _ = get_monthly_means(time, f_dswt)
    _, transport_m, _ = get_monthly_means(time, transport_contour)
    time_y, f_dswt_y, _ = get_yearly_means(time, f_dswt)
    _, transport_y, _ = get_yearly_means(time, transport_contour)
    
    fig = plt.figure(figsize=(6, 5))
    ax1 = plt.subplot(2, 1, 1)
    ax1 = plot_histogram_multiple_years(time_m, f_dswt_m*100,
                                        ylabel='Occurrence (%)',
                                        ylim = [0, 100], color=color_dswt,
                                        ax=ax1, show=False)
    ax1.plot(time_y, f_dswt_y*100, 'xk')
    ax1 = plot_yearly_grid(ax1, years)
    ax1.set_xlim(xlim)
    ax1.set_xticklabels([])
    ax1.set_yticks(np.arange(0, 120, 20))
    add_subtitle(ax1, '(a) DSWT monthly mean occurrence')
    
    # DSWT transport
    ax2 = plt.subplot(2, 1, 2)
    ylim2 = [0, np.round(np.nanmax(transport_m/(24*60*60))+0.4*np.nanmax(transport_m/(24*60*60)), 2)]
    ax2 = plot_histogram_multiple_years(time_m, transport_m/(24*60*60),
                                        ylabel='Transport (m$^2$ s$^{-1}$)',
                                        ylim=ylim2, color=color_transport,
                                        ax=ax2, show=False)
    ax2.plot(time_y, transport_y/(24*60*60), 'xk')
    plot_yearly_grid(ax2, years)
    ax2.plot(xlim, [0, 0], '-k')
    ax2.set_xlim(xlim)
    add_subtitle(ax2, '(b) DSWT monthly mean transport across 50 m contour')
    
    # save and show figure
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()
    
def plot_dswt_map(df_transport:pd.DataFrame,
                  grid_ds:xr.Dataset,
                  output_path=None,
                  show=True):
    # DSWT map
    l_time = np.ones(len(df_transport)).astype(bool) # mean map over all times
    transport_overall = get_transport_map(df_transport, l_time, grid_ds.lon_rho.shape)
    
    l_mask = grid_ds.h.values > 100.
    transport_overall[l_mask] = np.nan
    
    fig = plt.figure(figsize=(6, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    plot_basic_map(ax, lon_range_default, lat_range_default,
                   meridians_default, parallels_default, full_resolution=False)
    plot_contours(grid_ds.lon_rho.values, grid_ds.lat_rho.values, grid_ds.h.values,
                  lon_range_default, lat_range_default,
                  ax=ax, show=False, color='w',
                  clevels=[25, 50, 100, 200],
                  linewidths=[2.0, 4.0, 2.0, 2.0])
    plot_contours(grid_ds.lon_rho.values, grid_ds.lat_rho.values, grid_ds.h.values,
                  lon_range_default, lat_range_default,
                  ax=ax, show=False,
                  clevels=[25, 50, 100, 200],
                  linewidths=[1.0, 2.0, 1.0, 1.0])
    
    c = ax.pcolormesh(grid_ds.lon_rho.values, grid_ds.lat_rho.values, transport_overall/(24*60*60), cmap='viridis')
    add_subtitle(ax, 'Mean DSWT transport')
    
    # colorbar
    ll, bb, ww, hh = ax.get_position().bounds
    cax = fig.add_axes([ll+ww+0.02, bb, 0.02, hh])
    cbar = plt.colorbar(c, cax=cax)
    cbar.set_label('Transport (m$^2$ s$^{-1}$)')
    
    # save and show figure
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()

if __name__ == '__main__':
    dswt_output_dir = 'output/test_114-116E_33-31S/'
    years = [2017]
    grid_file = 'tests/data/grid.nc'
    depth_contour = 50
    
    grid_ds = xr.load_dataset(grid_file)
    
    df_transport = pd.read_csv(f'{dswt_output_dir}processed/dswt_transport_{years[0]}.csv')
    df_timeseries = pd.read_csv(f'{dswt_output_dir}processed/dswt_timeseries_{years[0]}.csv')

    plot_dswt_timeseries(df_timeseries, depth_contour, years, output_path='plots/test_timeseries.jpg')
    plot_dswt_map(df_transport, grid_ds, output_path='plots/test_map.jpg')