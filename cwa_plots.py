from readers.read_dswt_output import get_transport_map, read_dswt_transport, read_df_from_multiple_csvs
from readers.read_ocean_data import load_roms_data, select_roms_transect_from_known_coordinates, select_input_files, select_roms_subset
from readers.read_meteo_data import WindTimeseries
from readers.read_glider_data import GliderData, convert_glider_data_to_transect_data
from dswt.dswt_events import DswtEvents
from dswt.dswt_detection import determine_dswt_along_transect

from cwa_analyses import read_analyses_from_multiple_csvs, get_roms_contour_coordinates, get_roms_ds_along_contour

from transects import read_transects_in_lon_lat_range_from_json

from tools import log
from tools.files import get_dir_from_json, get_files_in_dir
from tools.timeseries import get_monthly_means, get_monthly_climatology, get_yearly_means, get_l_months, get_l_time_range, get_time_indices, add_month_to_time, get_closest_time_index
from tools.velocity_shore_angles import get_cross_and_along_shelf_velocities
from plot_tools.basic_timeseries import plot_histogram_multiple_years, plot_yearly_grid, plot_monthly_grid, plot_boxplots_multiple_years, plot_monthly_histogram, plot_multi_bar_monthly_histogram, plot_multi_bar_yearly_histogram
from plot_tools.general import add_subtitle, color_y_axis, get_vmin_vmax
from plot_tools.basic_maps import plot_basic_map, plot_contours, plot_bathymetry
from tools.coordinates import get_distance_between_points
from tools.roms import get_z
from tools.config import read_config

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import AnchoredText
import cartopy.crs as ccrs
import cmocean as cm
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
import os
import string
from scipy import stats

color_dswt = '#25419e'
color_transport = '#0e6e22'
color_pos = '#900C3F'
color_neg = '#1e1677'

color_dswt_std = "#6573a1"
color_transport_std = "#4cca67"
color_pos_std = "#E05389"
color_neg_std = "#6960d1"

lon_range_default = [114.5, 116.]
lat_range_default = [-33., -31.]
meridians_default = [115., 116.]
parallels_default = [-33., -32., -31.]

# Writing twice-daily means to reflect the land-seabreeze signal dominant in summer months
# from Rafiq et al. (2020) making the split according to the following times:
# - seabreeze (southerly winds) from 09:00 until 23:00 Perth local time (01:00 to 15:00 UTC)
# - landbreeze (easterly winds) from 23:00 until 08:00 Perth local time (15:00 to 00:00 UTC)
seabreeze_hour = 1
landbreeze_hour = 15
n_seabreeze_hours = 15
n_landbreeze_hours = 9

lim_southerly = [160, 200]
lim_northerly = [[0, 20], [340, 360]]
lim_onshore = [250, 290]
lim_offshore = [70, 110]

OMEGA = 7.292*10**-5 # rad/s
RHO0 = 1025.
LAT0 = -32.0
F = 2*OMEGA*np.sin(np.deg2rad(LAT0))
G = 9.81 # m2/s

# ---------------------------------------------------
# non-plotting functions
# ---------------------------------------------------
def _split_into_wind_dirs(df_analysis:pd.DataFrame):
    wind_dir = df_analysis['wind_dir'].values
    l_southerly = np.logical_and(wind_dir >= lim_southerly[0], wind_dir <= lim_southerly[1])
    l_northerly = np.logical_or(
        np.logical_and(wind_dir >= lim_northerly[0][0], wind_dir <= lim_northerly[0][1]),
        np.logical_and(wind_dir >= lim_northerly[1][0], wind_dir <= lim_northerly[1][1])
    )
    l_onshore = np.logical_and(wind_dir >= lim_onshore[0], wind_dir <= lim_onshore[1])
    l_offshore = np.logical_and(wind_dir >= lim_offshore[0], wind_dir <= lim_offshore[1])
    
    return l_southerly, l_northerly, l_onshore, l_offshore

def get_l_seabreeze_landbreeze(df_analysis:pd.DataFrame) -> tuple[np.ndarray[bool], np.ndarray[bool]]:
    hours = np.array([pd.to_datetime(d).hour for d in df_analysis['time'].values])
    l_sb = hours == seabreeze_hour
    l_lb = hours == landbreeze_hour
    return l_sb, l_lb   

def convert_df_to_daily_means(df_analysis:pd.DataFrame):
    days = np.array([pd.to_datetime(d).date() for d in df_analysis['time'].values])
    unique_days = np.unique(days)
    
    l_sb, l_lb = get_l_seabreeze_landbreeze(df_analysis)
    # multiply df by relevant hours
    df_lb = df_analysis.drop('time', axis=1)[l_lb] * n_landbreeze_hours
    df_sb = df_analysis.drop('time', axis=1)[l_sb] * n_seabreeze_hours
    # group by day and sum dfs
    df_lb['time'] = unique_days
    df_sb['time'] = unique_days
    
    df_daily = (pd.concat([df_lb, df_sb]).groupby(['time']).sum() / 24).reset_index()
    
    return df_daily

def _get_slope_estimate(grid_ds:xr.Dataset, transects_file='input/transects/cwa_transects.json', max_depth=100):
    transects = read_transects_in_lon_lat_range_from_json(transects_file, [114, 116], [-33, -31])
    lon = grid_ds.lon_rho.values
    lat = grid_ds.lat_rho.values
    slope = []
    for t in list(transects.keys()):
        etas = transects[t]['eta']
        xis = transects[t]['xi']
        h = grid_ds.h.values[etas, xis]
        dh = abs(h - max_depth)
        i = np.where(dh == min(dh))[0][0]
        distance = get_distance_between_points(lon[etas[i], xis[i]], lat[etas[i], xis[i]], lon[etas[0], xis[0]], lat[etas[0], xis[0]])
        slope.append((h[i] - h[0]) / distance)
    return np.nanmean(slope)

# ---------------------------------------------------
# Plots
# ---------------------------------------------------

# --- Introduction ---
def plot_overview_map(glider_ds:xr.Dataset, bathy_ds:xr.Dataset, global_dswt_df:pd.DataFrame,
                 output_path=None, show=False):
    
    aus_lon = [111.0, 129.0] # wa only
    aus_lat = [-36.0, -12.0]
    
    depth_ticks = [-150, -100, -50, 0]
    depth_ticklabels = [150, 100, 50, 0]
    depth_lim = [-150, 0]
    xlim = [glider_ds.distance.values[0]/1000, glider_ds.distance.values[-1]/1000]
    
    fig = plt.figure(figsize=(11, 11))
    plt.rcParams['font.size'] = 12
    plt.subplots_adjust(wspace=0.2)
    # --- Global map
    ax1 = plt.subplot(5, 3, (1, 8), projection=ccrs.Robinson())
    ax1.set_global()
    ax1.stock_img()
    ax1.coastlines()
    ax1.scatter(global_dswt_df['lon'].values, global_dswt_df['lat'].values, marker='^', s=40, c=color_pos, transform=ccrs.PlateCarree(), zorder=100)
    ax1.plot([aus_lon[0], aus_lon[1], aus_lon[1], aus_lon[0], aus_lon[0]],
             [aus_lat[0], aus_lat[0], aus_lat[1], aus_lat[1], aus_lat[0]],
             '-k', linewidth=1.0, transform=ccrs.PlateCarree(), zorder=99)
    
    anchored_text = AnchoredText('(a) DSWT around the world', loc='upper left',
                                 borderpad=0.0, bbox_to_anchor=(0.28, -0.04),
                                 bbox_transform=ax1.transAxes)
    anchored_text.zorder = 25
    ax1.add_artist(anchored_text)
    
    # --- WCS map
    ax2 = plt.subplot(5, 3, (3, 9), projection=ccrs.PlateCarree())
    plot_basic_map(ax2, lon_range_default, lat_range_default, meridians_default, parallels_default, full_resolution=True)
    plot_bathymetry(bathy_ds.lon.values, bathy_ds.lat.values, -bathy_ds.z.values,
                    lon_range_default, lat_range_default, ax=ax2, show=False,
                    vmin=0, vmax=300, cmap=cm.cm.deep)
    xx, yy = np.meshgrid(bathy_ds.lon.values, bathy_ds.lat.values)
    plot_contours(xx, yy, -bathy_ds.z.values,
                  lon_range_default, lat_range_default, ax=ax2, show=False,
                  clevels=[25, 50, 100, 200])
    plot_contours(xx, yy, -bathy_ds.z.values,
                  lon_range_default, lat_range_default, ax=ax2, show=False,
                  clevels=[1000], color='#e0e0e0')
    ax2.plot(glider_ds.lon.values, glider_ds.lat.values, '.w', linewidth=1.0, markersize=10)
    ax2.plot(glider_ds.lon.values, glider_ds.lat.values, '.k', linewidth=0.7, label='Ocean glider transect')
    ax2.legend(loc='lower right')
    
    add_subtitle(ax2, '(b) Wadjemup (Rottnest)\n      Continental Shelf')
    
    # --- Glider plots
    def _plot_glider_transect(ax:plt.axes, values:np.ndarray, vmin:float, vmax:float, cmap:str, cbar_label:str,
                              move_up=True):
        x = glider_ds.distance.values/1000
        c = ax.pcolormesh(x, glider_ds.z.values, np.fliplr(values), vmin=vmin, vmax=vmax, cmap=cmap)
        ax.fill_between(x, depth_lim[0], np.flip(glider_ds.h.values), color='#d2d2d2', edgecolor='k')
        ax.set_yticks(depth_ticks)
        ax.set_yticklabels(depth_ticklabels)
        ax.set_ylabel('Depth (m)')
        ax.set_xlabel('Distance along transect (km)')
        ax.set_xlim(xlim)
        ax.set_ylim(depth_lim)
        
        if vmin is None or vmax is None:
            min_bin = np.nanmin(values)
            max_bin = np.nanmax(values)
            dbin = (max_bin - min_bin) / 1000
            vmin, vmax = get_vmin_vmax(values, min_bin=min_bin, max_bin=max_bin, dbin=dbin)
        
        l, b, w, h = ax.get_position().bounds
        if move_up == True:
            ax.set_position([l, b+0.05, w, h])
        else:
            ax.set_position([l, b-0.08, w, h])
        l, b, w, h = ax.get_position().bounds
        cax = fig.add_axes([l, b-0.08, w, 0.02])
        cbar = plt.colorbar(c, cax=cax, orientation='horizontal')
        cbar.set_label(cbar_label)
        
    ax3 = plt.subplot(5, 3, 10)
    _plot_glider_transect(ax3, glider_ds.density.values-1000, 24.8, 25.6, cm.cm.thermal_r, '$\sigma_T$ (kg m$^{-3}$)')
    add_subtitle(ax3, '(c) Density', location='lower right')
    
    ax4 = plt.subplot(5, 3, 11)
    _plot_glider_transect(ax4, glider_ds.temp.values, 20, 22, 'RdYlBu_r', 'Temperature ($^o$C)')
    ax4.set_yticklabels([])
    ax4.set_ylabel('')
    add_subtitle(ax4, '(d) Temperature', location='lower right')
    
    ax5 = plt.subplot(5, 3, 12)
    _plot_glider_transect(ax5, glider_ds.salt.values, 35.7, 35.9, cm.cm.haline, 'Salinity')
    ax5.set_yticklabels([])
    ax5.set_ylabel('')
    add_subtitle(ax5, '(e) Salinity', location='lower right')
    
    ax6 = plt.subplot(5, 3, 13)
    _plot_glider_transect(ax6, glider_ds.ox2.values, 180, 190, cm.cm.tempo, 'Dissolved oxygen ($\mu$mol kg$^{-1}$)', move_up=False)
    add_subtitle(ax6, '(f) Dissolved O$_2$', location='lower right')
    
    ax7 = plt.subplot(5, 3, 14)
    _plot_glider_transect(ax7, glider_ds.cphl.values, 0.5, 1.2, 'summer', 'Chlorophyll\n(mg $m$^{-3}$)', move_up=False)
    ax7.set_yticklabels([])
    ax7.set_ylabel('')
    add_subtitle(ax7, '(g) Chlorophyll', location='lower right')
    
    ax8 = plt.subplot(5, 3, 15)
    _plot_glider_transect(ax8, glider_ds.bbp.values*10**3, 0, 13, cm.cm.turbid, 'Backscatter (10$^{-3}$ m$^{-1}$)', move_up=False)
    ax8.set_yticklabels([])
    ax8.set_ylabel('')
    add_subtitle(ax8, '(h) Backscatter', location='lower right')
    
    # move ax1
    l1, b1, w1, h1 = ax1.get_position().bounds
    ax1.set_position([l1-0.05, b1, w1, h1])
    
    # add AUS map
    ax_aus = fig.add_axes([l1+0.75*w1, b1-0.04, 0.3*w1, 0.35*h1], projection=ccrs.PlateCarree())
    ax_aus.stock_img()
    ax_aus.coastlines()
    ax_aus.set_extent([aus_lon[0], aus_lon[1], aus_lat[0], aus_lat[1]], ccrs.PlateCarree())
    ax_aus.plot([lon_range_default[0], lon_range_default[1], lon_range_default[1], lon_range_default[0], lon_range_default[0]],
                [lat_range_default[0], lat_range_default[0], lat_range_default[1], lat_range_default[1], lat_range_default[0]],
                '-w', linewidth=3)
    ax_aus.plot([lon_range_default[0], lon_range_default[1], lon_range_default[1], lon_range_default[0], lon_range_default[0]],
                [lat_range_default[0], lat_range_default[0], lat_range_default[1], lat_range_default[1], lat_range_default[0]],
                '-k', linewidth=2)
    
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()

# --- Results ---
# --- General cross-shelf transport
def plot_u_bar_overview(ucross_ds:xr.Dataset,
                   cmap='RdYlBu_r', vmin=-0.1, vmax=0.1,
                   output_path=None, show=False):
    
    lon_range = [114.0, 116.0]
    lat_range = [-33.0, -31.0]
    grid_ds = xr.load_dataset(f'{get_dir_from_json("cwa")}grid.nc')
    grid_ds = select_roms_subset(grid_ds, time_range=None, lon_range=lon_range, lat_range=lat_range)
    
    ocean_time = np.array([pd.to_datetime(d) for d in ucross_ds['ocean_time'].values])
    
    ubar = np.nanmean(ucross_ds['u_bar'].values, axis=0)
    ubar[ubar == 0] = np.nan
    
    lon_contour50, lat_contour50, _ = get_roms_contour_coordinates(grid_ds, lon_range, lat_range, 50)
    ucross_ds_contour50 = get_roms_ds_along_contour(ucross_ds, grid_ds, lon_contour50, lat_contour50)
    
    lon_contour100, lat_contour100, _ = get_roms_contour_coordinates(grid_ds, lon_range, lat_range, 100)
    ucross_ds_contour100 = get_roms_ds_along_contour(ucross_ds, grid_ds, lon_contour100, lat_contour100)
    
    lon_contour200, lat_contour200, _ = get_roms_contour_coordinates(grid_ds, lon_range, lat_range, 200)
    ucross_ds_contour200 = get_roms_ds_along_contour(ucross_ds, grid_ds, lon_contour200, lat_contour200)
    
    time_m, ubar50, ubar50_std = get_monthly_means(ocean_time, np.nanmean(ucross_ds_contour50['u_bar'].values, axis=1))
    _, ubar100, ubar100_std = get_monthly_means(ocean_time, np.nanmean(ucross_ds_contour100['u_bar'].values, axis=1))
    _, ubar200, ubar200_std = get_monthly_means(ocean_time, np.nanmean(ucross_ds_contour200['u_bar'].values, axis=1))
    
    # ubar50_sv = np.nansum(ucross_ds_contour50['u_bar'].values * ucross_ds_contour50['dx'].values * ucross_ds_contour50['h'].values, axis=1) / 10**6
    # ubar100_sv = np.nansum(ucross_ds_contour100['u_bar'].values * ucross_ds_contour100['dx'].values * ucross_ds_contour100['h'].values, axis=1) / 10**6
    # ubar200_sv = np.nansum(ucross_ds_contour200['u_bar'].values * ucross_ds_contour200['dx'].values * ucross_ds_contour200['h'].values, axis=1) / 10**6
    
    # _, ubar50_sv_m, ubar50_sv_std = get_monthly_means(ocean_time, ubar50_sv)
    # _, ubar100_sv_m, ubar100_sv_std = get_monthly_means(ocean_time, ubar100_sv)
    # _, ubar200_sv_m, ubar200_sv_std = get_monthly_means(ocean_time, ubar200_sv)
    
    xlim = [ocean_time[0], ocean_time[-1]]
    ylim = [-0.1, 0.1]
    ylim_sv = [-2.0, 2.0]
    
    fig = plt.figure(figsize=(10, 5))
    
    ax1 = plt.subplot(3, 3, (1, 2))
    plot_monthly_histogram(time_m, ubar50, yerr=ubar50_std, err_color=color_neg_std,
                           color=color_neg, ylim=ylim, ax=ax1, show=False, time_is_center=True)
    ax1.plot(xlim, [0, 0], '-k')
    ax1.set_xticklabels([])
    ax1.set_ylabel(r'$\bar{u}$ (m s$^{-1}$)')
    ax1.set_xlim(xlim)
    add_subtitle(ax1, '(a) Depth-mean cross-shelf velocity across 50 m')
    
    # ax11 = ax1.twinx()
    # ax11.vlines(time_m, ubar50_sv_m-ubar50_sv_std, ubar50_sv_m+ubar50_sv_std, colors='#808080')
    # ax11.scatter(time_m, ubar50_sv_m, marker='x', c='k', s=10)
    # ax11.set_ylim(ylim_sv)
    # ax11.set_ylabel(r'$\bar{U}$ (Sv)')
    
    ax2 = plt.subplot(3, 3, (4, 5))
    plot_monthly_histogram(time_m, ubar100, yerr=ubar100_std, err_color=color_neg_std,
                           color=color_neg, ylim=ylim, ax=ax2, show=False, time_is_center=True)
    ax2.plot(xlim, [0, 0], '-k')
    ax2.set_xticklabels([])
    ax2.set_ylabel(r'$\bar{u}$ (m s$^{-1}$)')
    ax2.set_xlim(xlim)
    add_subtitle(ax2, '(b) Depth-mean cross-shelf velocity across 100 m')
    
    # ax22 = ax2.twinx()
    # ax22.vlines(time_m, ubar100_sv_m-ubar100_sv_std, ubar100_sv_m+ubar100_sv_std, colors='#808080')
    # ax22.scatter(time_m, ubar100_sv_m, marker='x', c='k', s=10)
    # ax22.set_ylim(ylim_sv)
    # ax22.set_ylabel(r'$\bar{U}$ (Sv)')
    
    ax3 = plt.subplot(3, 3, (7, 8))
    plot_monthly_histogram(time_m, ubar200, yerr=ubar200_std, err_color=color_neg_std,
                           color=color_neg, ylim=ylim, ax=ax3, show=False, time_is_center=True)
    ax3.plot(xlim, [0, 0], '-k')
    ax3.set_ylabel(r'$\bar{u}$ (m s$^{-1}$)')
    ax3.set_xlim(xlim)
    add_subtitle(ax3, '(c) Depth-mean cross-shelf velocity across 200 m')
    
    # ax33 = ax3.twinx()
    # ax33.vlines(time_m, ubar200_sv_m-ubar200_sv_std, ubar200_sv_m+ubar200_sv_std, colors='#808080')
    # ax33.scatter(time_m, ubar200_sv_m, marker='x', c='k', s=10)
    # ax33.set_ylim(ylim_sv)
    # ax33.set_ylabel(r'$\bar{U}$ (Sv)')
    
    ax4 = plt.subplot(3, 3, (3, 9), projection=ccrs.PlateCarree())
    plot_basic_map(ax4, lon_range_default, lat_range_default,
                   meridians=meridians_default, parallels=parallels_default)
    c = ax4.pcolormesh(ucross_ds.lon_rho.values, ucross_ds.lat_rho.values, ubar, cmap=cmap, vmin=vmin, vmax=vmax)
    plot_contours(ucross_ds.lon_rho.values, ucross_ds.lat_rho.values, ucross_ds.h.values,
                  lon_range_default, lat_range_default,
                  ax=ax4, show=False,
                  clevels=[50, 100, 200],
                  linewidths=[2.0, 1.0, 1.0])
    add_subtitle(ax4, '(d) Depth-mean cross-shelf velocity')
    
    # rescale map
    _, b1, _, h1 = ax1.get_position().bounds
    l3, b3, w3, h3 = ax3.get_position().bounds
    l4, _, w4, h4 = ax4.get_position().bounds
    
    ax4.set_position([l4+0.02, b3, w4/h4*(b1+h1-b3), b1+h1-b3])
    
    # colorbar
    l4n, b4n, w4n, h4n = ax4.get_position().bounds
    cax = fig.add_axes([l4n+w4n+0.02, b4n, 0.02, h4n])
    cbar = plt.colorbar(c, cax=cax)
    cbar.set_label('Depth mean cross-shelf velocity (m s$^{-1}$)')
    
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()

def plot_u_bar_seasonality_maps(ucross_ds:xr.Dataset,
                   cmap='RdYlBu_r', vmin=-0.2, vmax=0.2,
                   output_path=None, show=False):
    
    lon_range = [114.0, 116.0]
    lat_range = [-33.0, -31.0]
    grid_ds = xr.load_dataset(f'{get_dir_from_json("cwa")}grid.nc')
    grid_ds = select_roms_subset(grid_ds, time_range=None, lon_range=lon_range, lat_range=lat_range)
    
    ocean_time = np.array([pd.to_datetime(d) for d in ucross_ds['ocean_time'].values])
    l_jan = get_l_time_range(ocean_time, datetime(2017, 1, 1), datetime(2017, 1, 31))
    l_jun = get_l_time_range(ocean_time, datetime(2017, 6, 1), datetime(2017, 6, 30))
    
    ubar_jan = np.nanmean(ucross_ds['u_bar'].values[l_jan, :, :], axis=0)
    ubar_jan[ubar_jan == 0] = np.nan
    ubar_jun = np.nanmean(ucross_ds['u_bar'].values[l_jun, :, :], axis=0)
    ubar_jun[ubar_jun == 0] = np.nan
    
    fig = plt.figure(figsize=(8, 6))
    
    ax4 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
    plot_basic_map(ax4, lon_range_default, lat_range_default,
                   meridians=meridians_default, parallels=parallels_default)
    c = ax4.pcolormesh(ucross_ds.lon_rho.values, ucross_ds.lat_rho.values, ubar_jan, cmap=cmap, vmin=vmin, vmax=vmax)
    plot_contours(ucross_ds.lon_rho.values, ucross_ds.lat_rho.values, ucross_ds.h.values,
                  lon_range_default, lat_range_default,
                  ax=ax4, show=False,
                  clevels=[50, 100, 200, 1000],
                  linewidths=[2.0, 1.0, 1.0, 1.0])
    add_subtitle(ax4, r'(a) January $\bar{u}$', location='upper right')
    
    ax5 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
    plot_basic_map(ax5, lon_range_default, lat_range_default,
                   meridians=meridians_default, parallels=parallels_default)
    c = ax5.pcolormesh(ucross_ds.lon_rho.values, ucross_ds.lat_rho.values, ubar_jun, cmap=cmap, vmin=vmin, vmax=vmax)
    plot_contours(ucross_ds.lon_rho.values, ucross_ds.lat_rho.values, ucross_ds.h.values,
                  lon_range_default, lat_range_default,
                  ax=ax5, show=False,
                  clevels=[50, 100, 200, 1000],
                  linewidths=[2.0, 1.0, 1.0, 1.0])
    ax5.set_yticklabels([])
    add_subtitle(ax5, r'(b) June $\bar{u}$', location='upper right')
    
    # colorbar
    l5n, b5n, w5n, h5n = ax5.get_position().bounds
    cax = fig.add_axes([l5n+w5n+0.02, b5n, 0.02, h5n])
    cbar = plt.colorbar(c, cax=cax)
    cbar.set_label('Depth mean cross-shelf velocity (m s$^{-1}$)')
    
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()

def plot_u_prime_evolution(ucross_ds:xr.Dataset, df_analysis:pd.DataFrame,
                           cmap='RdYlBu_r', vmin=-0.1, vmax=0.1,
                           output_path=None, show=False):
    lon_range = [114.0, 116.0]
    lat_range = [-33.0, -31.0]
    grid_ds = xr.load_dataset(f'{get_dir_from_json("cwa")}grid.nc')
    grid_ds = select_roms_subset(grid_ds, time_range=None, lon_range=lon_range, lat_range=lat_range)
    
    lon_contour50, lat_contour50, _ = get_roms_contour_coordinates(grid_ds, lon_range, lat_range, 50)
    ucross_ds_contour50 = get_roms_ds_along_contour(ucross_ds, grid_ds, lon_contour50, lat_contour50)
    
    u_prime50 = np.nanmean(ucross_ds_contour50['u_prime'].values, axis=2) # [time, depth, distance]
    u50 = np.nanmean(ucross_ds_contour50['u_cross'].values, axis=2) # [time, depth, distance]
    ubar50 = np.nanmean(ucross_ds_contour50['u_bar'].values, axis=1) # [time, distance]
    z_rho50 = np.nanmean(ucross_ds_contour50['z_rho'].values, axis=1)
    lvc_50 = np.nanmean(ucross_ds_contour50['L_vc'].values, axis=1)

    df_analysis = convert_df_to_daily_means(df_analysis)
    df_time = df_analysis['time'].values
    z_surface = -df_analysis['zss'].values
    z_bottom = -50 + df_analysis['zsb'].values

    xlim = [ucross_ds.ocean_time.values[0], ucross_ds.ocean_time.values[-1]]
    center_times = np.array([datetime(2017, m, 15) for m in np.arange(1, 13)])
    center_times_str = np.array([m.strftime('%b') for m in center_times])
    
    fig = plt.figure(figsize=(8, 6))
    
    ax2 = plt.subplot(5, 1, 1)
    ax2.plot(ucross_ds.ocean_time.values, ubar50, '-k', linewidth=1.0)
    ax2.plot(xlim, [0, 0], '-k', linewidth=0.5)
    ax2.set_ylim([-0.06, 0.1])
    ax2.set_yticks([-0.06, -0.03, 0, 0.03, 0.06])
    plot_monthly_grid(ax2, 2017, color="#666666", alpha=1.0, linewidth=0.7)
    ax2.set_xlim(xlim)
    ax2.set_xticks(center_times)
    ax2.set_xticklabels([])
    ax2.set_ylabel(r'$\bar{u}$ (m s$^{-1}$)')
    add_subtitle(ax2, '(a) Depth-mean cross-shelf velocity across 50 m')
    
    ax1 = plt.subplot(5, 1, (2, 3))
    c = ax1.pcolormesh(ucross_ds.ocean_time.values, z_rho50, u_prime50.transpose(), cmap=cmap, vmin=vmin, vmax=vmax)
    plot_monthly_grid(ax1, 2017, color="#666666", alpha=1.0, linewidth=0.7)
    ax1.set_yticks([-50, -40, -30, -20, -10, 0])
    ax1.set_ylabel('Depth (m)')
    ax1.set_xticks(center_times)
    ax1.set_xticklabels([])
    ax1.set_xlim(xlim)
    add_subtitle(ax1, "(b) Cross-shelf velocity variability u' across 50 m", alpha=0.5)
    
    ax3 = plt.subplot(5, 1, (4, 5))
    c = ax3.pcolormesh(ucross_ds.ocean_time.values, z_rho50, u50.transpose(), cmap=cmap, vmin=vmin, vmax=vmax)
    plot_monthly_grid(ax3, 2017, color="#666666", alpha=1.0, linewidth=0.7)
    ax3.set_yticks([-50, -40, -30, -20, -10, 0])
    ax3.set_ylabel('Depth (m)')
    ax3.set_xticks(center_times)
    ax3.set_xticklabels(center_times_str)
    ax3.set_xlim(xlim)
    add_subtitle(ax3, '(c) Cross-shelf velocity u across 50 m', alpha=0.5)
    
    ax3.plot(df_time, z_surface, '-k', linewidth=0.5)
    
    # colorbar
    l, b, w, h = ax3.get_position().bounds
    cax = fig.add_axes([l, b-0.1, w, 0.04])
    cbar = plt.colorbar(c, cax=cax, orientation='horizontal')
    cbar.set_label("(m s$^{-1}$)")
    
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()

def plot_us_ub_dynamics(df_analysis:pd.DataFrame, output_path=None, show=False):
    df_analysis_daily = convert_df_to_daily_means(df_analysis) # needed because df_analysis split into seabreeze and landbreeze
    time = np.array([pd.to_datetime(t) for t in df_analysis_daily['time'].values])
    us = df_analysis_daily['Uss'].values
    ub = df_analysis_daily['Usb'].values
    ue = df_analysis_daily['Tes'].values
    
    l_southerly, l_northerly, l_onshore, l_offshore = _split_into_wind_dirs(df_analysis)
    df_southerly = df_analysis.loc[l_southerly]
    df_northerly = df_analysis.loc[l_northerly]
    df_onshore = df_analysis.loc[l_onshore]
    df_offshore = df_analysis.loc[l_offshore]
    time_southerly = np.array([pd.to_datetime(d) for d in df_southerly['time'].values])
    time_northerly = np.array([pd.to_datetime(d) for d in df_northerly['time'].values])
    time_onshore = np.array([pd.to_datetime(d) for d in df_onshore['time'].values])
    time_offshore = np.array([pd.to_datetime(d) for d in df_offshore['time'].values])
    
    time_m, us_m, us_std = get_monthly_means(time, us)
    _, ue_m, ue_std = get_monthly_means(time, ue)
    _, ub_m, ub_std = get_monthly_means(time, ub)
    
    xlim = [time[0], time[-1]]
    center_times = np.array([datetime(2017, m, 15) for m in np.arange(1, 13)])
    center_times_str = np.array([m.strftime("%b") for m in center_times])
    
    color_summer = "#D19E3132"
    color_winter = "#483f9932"
    
    fig = plt.figure(figsize=(11, 8))
    
    # --- timeseries ---
    # monthly surface timeseries
    ax1 = plt.subplot(3, 5, (1, 3))
    plot_multi_bar_monthly_histogram(time_m, [us_m, ue_m], [color_pos, "#9B9B9B"], ['U$_s$', 'U$_{E, s}$'],
                                             ylim=[-1.0, 1.4], ylabel='Transport (m$^2$ s$^{-1}$)', legend_loc='lower left',
                                             ax=ax1, show=False)
    ax1.set_xlim(xlim)
    ax1.plot(xlim, [0, 0], '-k')
    plot_monthly_grid(ax1, 2017)
    ax1.set_xticks(center_times)
    ax1.set_xticklabels([])
    add_subtitle(ax1, '(a) Surface cross-shelf transport')
    ax1.axvspan(datetime(2017, 1, 1), datetime(2017, 1, 31), color=color_summer, zorder=0)
    ax1.axvspan(datetime(2017, 12, 1), datetime(2017, 12, 31), color=color_summer,zorder=0)
    ax1.axvspan(datetime(2017, 5, 1), datetime(2017, 7, 31), color=color_winter, zorder=0)
    
    ax1.text(1.01, 0.2, 'offshore', rotation=90, va='center', transform=ax1.transAxes)
    ax1.text(1.01, 0.7, 'onshore', rotation=90, va='center', transform=ax1.transAxes)

    # monthly bottom timeseries
    ax2 = plt.subplot(3, 5, (6, 8))
    plot_monthly_histogram(time_m, ub_m, color=color_neg, ylim=[-0.4, 0.4], ylabel='Transport (m$^2$ s$^{-1}$)',
                           time_is_center=True, ax=ax2, show=False)
    ax2.set_xlim(xlim)
    ax2.plot(xlim, [0, 0], '-k')
    plot_monthly_grid(ax2, 2017)
    ax2.set_xticks(center_times)
    ax2.set_xticklabels(center_times_str)
    add_subtitle(ax2, '(b) Bottom cross-shelf transport')
    ax2.axvspan(datetime(2017, 1, 1), datetime(2017, 1, 31), color=color_summer, zorder=0)
    ax2.axvspan(datetime(2017, 12, 1), datetime(2017, 12, 31), color=color_summer, zorder=0)
    ax2.axvspan(datetime(2017, 5, 1), datetime(2017, 7, 31), color=color_winter, zorder=0)
    
    ax2.text(1.01, 0.25, 'offshore', rotation=90, va='center', transform=ax2.transAxes)
    ax2.text(1.01, 0.75, 'onshore', rotation=90, va='center', transform=ax2.transAxes)
    
    # --- u versus wind ----
    xlim_wv = [0, 15]
    ylim_t = [-3.0, 3.0]
    
    # summer upwelling favorable (southerly)
    ax3 = plt.subplot(3, 5, 4)
    l_dj_s = np.logical_or(get_l_time_range(time_southerly, datetime(2017, 1, 1), datetime(2017, 1, 31)),
                           get_l_time_range(time_southerly, datetime(2017, 12, 1), datetime(2017, 12, 31)))
    ax3.scatter(df_southerly['wind_vel'].values[l_dj_s], df_southerly['Uss'].values[l_dj_s], marker='x', s=20, color=color_pos, label='U$_s$')
    ax3.scatter(df_southerly['wind_vel'].values[l_dj_s], df_southerly['Tes'].values[l_dj_s], marker='x', s=20, color='#9B9B9B', label='U_${E, s}$')
    ax3.scatter(df_southerly['wind_vel'].values[l_dj_s], df_southerly['Usb'].values[l_dj_s], marker='o', s=20, color=color_neg, label='U$_b$')
    ax3.plot(xlim_wv, [0, 0], '-k')
    ax3.set_xlim(xlim_wv)
    ax3.set_ylim(ylim_t)
    ax3.set_ylabel('Transport (m$^2$ s$^{-1}$)')
    ax3.set_xticklabels([])
    add_subtitle(ax3, '(c) Upwelling')
    ax3.set_facecolor(color_summer)
    
    # summer offshore
    ax4 = plt.subplot(3, 5, 5)
    l_dj_off = np.logical_or(get_l_time_range(time_offshore, datetime(2017, 1, 1), datetime(2017, 1, 31)),
                           get_l_time_range(time_offshore, datetime(2017, 12, 1), datetime(2017, 12, 31)))
    ax4.scatter(df_offshore['wind_vel'].values[l_dj_off], df_offshore['Uss'].values[l_dj_off], marker='x', s=20, color=color_pos, label='U$_s$')
    ax4.scatter(df_offshore['wind_vel'].values[l_dj_off], df_offshore['Usb'].values[l_dj_off], marker='o', s=20, color=color_neg, label='U$_b$')
    ax4.scatter([], [], marker='x', s=20, color='#9B9B9B', label='U$_{E, s}$')
    ax4.plot(xlim_wv, [0, 0], '-k')
    ax4.set_xlim(xlim_wv)
    ax4.set_ylim(ylim_t)
    ax4.set_yticklabels([])
    ax4.set_xticklabels([])
    add_subtitle(ax4, '(d) Offshore')
    ax4.set_facecolor(color_summer)
    
    ax4.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))
    
    # winter upwelling
    ax5 = plt.subplot(3, 5, 9)
    l_mjj_s = get_l_time_range(time_southerly, datetime(2017, 5, 1), datetime(2017, 7, 31))
    ax5.scatter(df_southerly['wind_vel'].values[l_mjj_s], df_southerly['Uss'].values[l_mjj_s], marker='x', s=20, color=color_pos, label='U$_s$')
    ax5.scatter(df_southerly['wind_vel'].values[l_mjj_s], df_southerly['Tes'].values[l_mjj_s], marker='x', s=20, color='#9B9B9B', label='U_${E, s}$')
    ax5.scatter(df_southerly['wind_vel'].values[l_mjj_s], df_southerly['Usb'].values[l_mjj_s], marker='o', s=20, color=color_neg, label='U$_b$')
    ax5.plot(xlim_wv, [0, 0], '-k')
    ax5.set_xlim(xlim_wv)
    ax5.set_ylim(ylim_t)
    ax5.set_ylabel('Transport (m$^2$ s$^{-1}$)')
    ax5.set_xticklabels([])
    add_subtitle(ax5, '(e) Upwelling')
    ax5.set_facecolor(color_winter)
    
    # winter downwelling
    ax6 = plt.subplot(3, 5, 10)
    l_mjj_n = get_l_time_range(time_northerly, datetime(2017, 5, 1), datetime(2017, 7, 31))
    ax6.scatter(df_northerly['wind_vel'].values[l_mjj_n], df_northerly['Uss'].values[l_mjj_n], marker='x', s=20, color=color_pos, label='U$_s$')
    ax6.scatter(df_northerly['wind_vel'].values[l_mjj_n], df_northerly['Tes'].values[l_mjj_n], marker='x', s=20, color='#9B9B9B', label='U_${E, s}$')
    ax6.scatter(df_northerly['wind_vel'].values[l_mjj_n], df_northerly['Usb'].values[l_mjj_n], marker='o', s=20, color=color_neg, label='U$_b$')
    ax6.plot(xlim_wv, [0, 0], '-k')
    ax6.set_xlim(xlim_wv)
    ax6.set_ylim(ylim_t)
    ax6.set_yticklabels([])
    ax6.set_xticklabels([])
    add_subtitle(ax6, '(f) Downwelling')
    ax6.set_facecolor(color_winter)
    
    # winter onshore
    ax7 = plt.subplot(3, 5, 14)
    l_mjj_on = get_l_time_range(time_onshore, datetime(2017, 5, 1), datetime(2017, 7, 31))
    ax7.scatter(df_onshore['wind_vel'].values[l_mjj_on], df_onshore['Uss'].values[l_mjj_on], marker='x', s=20, color=color_pos, label='U$_s$')
    ax7.scatter(df_onshore['wind_vel'].values[l_mjj_on], df_onshore['Usb'].values[l_mjj_on], marker='o', s=20, color=color_neg, label='U$_b$')
    ax7.plot(xlim_wv, [0, 0], '-k')
    ax7.set_xlim(xlim_wv)
    ax7.set_ylim(ylim_t)
    ax7.set_ylabel('Transport (m$^2$ s$^{-1}$)')
    ax7.set_xlabel('Wind speed (m s$^{-1}$)')
    add_subtitle(ax7, '(g) Onshore')
    ax7.set_facecolor(color_winter)
    
    # winter offshore
    ax8 = plt.subplot(3, 5, 15)
    l_mjj_off = get_l_time_range(time_offshore, datetime(2017, 5, 1), datetime(2017, 7, 31))
    ax8.scatter(df_offshore['wind_vel'].values[l_mjj_off], df_offshore['Uss'].values[l_mjj_off], marker='x', s=20, color=color_pos, label='U$_s$')
    ax8.scatter(df_offshore['wind_vel'].values[l_mjj_off], df_offshore['Usb'].values[l_mjj_off], marker='o', s=20, color=color_neg, label='U$_b$')
    ax8.plot(xlim_wv, [0, 0], '-k')
    ax8.set_xlim(xlim_wv)
    ax8.set_ylim(ylim_t)
    ax8.set_yticklabels([])
    ax8.set_xlabel('Wind speed (m s$^{-1}$)')
    add_subtitle(ax8, '(h) Offshore')
    ax8.set_facecolor(color_winter)
    
    # move axes
    l1, b1, w1, h1 = ax1.get_position().bounds
    l2, b2, w2, h2 = ax2.get_position().bounds
    
    l3, b3, w3, h3 = ax3.get_position().bounds
    l5, b5, w5, h5 = ax5.get_position().bounds
    l7, b7, w7, h7 = ax7.get_position().bounds
    
    ax1.set_position([l1-0.05, b5+0.5*h5+0.02, w1, b3+0.5*h3-b5-0.02])
    ax2.set_position([l2-0.05, b7, w2, b5+0.5*h5-b7-0.02])
    
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()

# --- DSWT
def plot_overall_transport_map_with_monthly_climatology(df_timeseries:pd.DataFrame,
                                                        df_analysis:pd.DataFrame,
                                                        df_transport:pd.DataFrame,
                                                        grid_ds:xr.Dataset,
                                                        output_path=None,
                                                        show=False):
    # DSWT climatology data
    months = np.arange(1, 13)
    time_m = []
    for m in months:
        time_m.append(datetime(2000, m, 15))
    time_m = np.array(time_m)
    xtick_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    
    xlim = [datetime(2000, 1, 1), datetime(2000, 12, 31)]
    
    time = np.array([pd.to_datetime(d) for d in df_timeseries['time'].values])
    dswt_transport_m, dswt_transport_std = get_monthly_climatology(time, df_timeseries['transport_50m'].values)
    
    # surface buoyancy flux climatology data
    df_analysis = convert_df_to_daily_means(df_analysis)
    time_b = np.array([pd.to_datetime(d) for d in df_analysis['time'].values])
    bflux_m, bflux_std = get_monthly_climatology(time_b, df_analysis['bflux_sh'].values)
    
    # DSWT map data
    # l_time = np.ones(len(df_transport)).astype(bool) # mean map over all times
    time_ds = np.array([pd.to_datetime(d) for d in df_transport['time'].values])
    l_months = get_l_months(time_ds, [5, 6, 7])
    transport_overall = get_transport_map(df_transport, l_months, grid_ds.lon_rho.shape)
    
    l_mask = grid_ds.h.values > 100.
    transport_overall[l_mask] = np.nan
    transport_overall[transport_overall == 0] = np.nan
    
    # --- figure ---
    fig = plt.figure(figsize=(8, 5))
    plt.subplots_adjust(hspace=0.1)
    
    # timeseries
    ax1 = plt.subplot(2, 3, (1, 2))
    plot_monthly_histogram(time_m, dswt_transport_m/(24*60*60), yerr=dswt_transport_std/(24*60*60),
                           err_color=color_transport_std, color=color_transport, time_is_center=True,
                           ylabel='Transport m$^2$ s$^{-1}$',
                           ylim=[0, 0.8],
                           ax=ax1, show=False)
    ax1.set_xlim(xlim)
    ax1.set_xticks(time_m)
    ax1.set_xticklabels([])
    add_subtitle(ax1, '(a) DSWT transport climatology')
    
    ax2 = plt.subplot(2, 3, (4, 5))
    plot_monthly_histogram(time_m, bflux_m*10**5, yerr=bflux_std*10**5,
                           c_change=0, color=[color_neg, color_pos], err_color=[color_neg_std, color_pos_std],
                           time_is_center=True, ylabel='Buoyancy flux\n(10$^{-5}$ kg s$^{-1}$ m$^{-2}$)',
                           ylim=[-2.0, 2.0], ax=ax2, show=False)
    ax2.plot(xlim, [0, 0], '-k')
    ax2.set_xlim(xlim)
    ax2.set_xticks(time_m)
    ax2.set_xticklabels(xtick_labels)
    add_subtitle(ax2, '(b) Inshore surface buoyancy flux climatology')
    
    # map inset
    l2, b2, w2, h2 = ax2.get_position().bounds
    ax22 = fig.add_axes([l2+0.01*w2, b2+0.02*h2, 0.15*w2, 0.3*h2], projection=ccrs.PlateCarree())
    plot_basic_map(ax22, lon_range_default, lat_range_default)
    deep_shallow = np.empty(grid_ds.h.shape)*np.nan
    l_shallow = grid_ds.h <= 20.
    deep_shallow[l_shallow] = 0
    cmap = ListedColormap(['#9794be'])#, '#d7789d'])
    ax22.pcolormesh(grid_ds.lon_rho.values, grid_ds.lat_rho.values, deep_shallow, vmin=0, vmax=1, cmap=cmap, zorder=1)
    
    # map
    ax3 = plt.subplot(2, 3, (3, 6), projection=ccrs.PlateCarree())
    
    plot_basic_map(ax3, lon_range_default, lat_range_default,
                   meridians_default, parallels_default, full_resolution=False)
    plot_contours(grid_ds.lon_rho.values, grid_ds.lat_rho.values, grid_ds.h.values,
                  lon_range_default, lat_range_default,
                  ax=ax3, show=False, color='w',
                  clevels=[25, 50, 100, 200],
                  linewidths=[2.0, 4.0, 2.0, 2.0])
    plot_contours(grid_ds.lon_rho.values, grid_ds.lat_rho.values, grid_ds.h.values,
                  lon_range_default, lat_range_default,
                  ax=ax3, show=False,
                  clevels=[25, 50, 100, 200],
                  linewidths=[1.0, 2.0, 1.0, 1.0])
    
    c = ax3.pcolormesh(grid_ds.lon_rho.values, grid_ds.lat_rho.values, transport_overall/(24*60*60), cmap='viridis', vmin=0, vmax=0.6)
    add_subtitle(ax3, '(c) Mean DSWT transport')
    
    # move axis
    l1, b1, w1, h1 = ax1.get_position().bounds
    l2, b2, w2, h2 = ax2.get_position().bounds
    l3, b3, w3, h3 = ax3.get_position().bounds
    ax3.set_position([l3+0.05, b2, w3/h3*(b1+h1-b2), b1+h1-b2])
    
    # colorbar
    ll, bb, ww, hh = ax3.get_position().bounds
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

def plot_dswt_forcing(df_dswt:pd.DataFrame, df_analysis:pd.DataFrame, grid_ds:xr.Dataset, output_path=None, show=False):
    df_analysis = convert_df_to_daily_means(df_analysis)
    
    time = np.array([pd.to_datetime(d) for d in df_dswt['time'].values])
    l_filter = get_l_months(time, [5, 6, 7])
    df_analysis = df_analysis.loc[l_filter]
    df_dswt = df_dswt[l_filter]
    
    bhflux = df_analysis['bflux_sh'].values * 10**5
    dsst = df_analysis['sst_sh'].values - df_analysis['sst_dp'].values
    drhodx = df_dswt['drhodx'].values * 10**5
    transport = df_dswt['transport_50m'].values / (24*60*60)
    vel = df_dswt['transport_50m'].values / df_dswt['thickness_50m'].values / (24*60*60)
    
    n_samples = len(vel)
    n_bins = 30
    vmin = 0
    vmax = 0.015
    
    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(wspace=0.3)
    
    ax1 = plt.subplot(2, 2, 1)
    # ax1.scatter(df_analysis['bflux_sh'].values * 10**5, df_dswt['drhodx'].values * 10**5, marker='x', s=20, c=color_neg)
    density1, bhflux_bins, drhodx_bins = np.histogram2d(bhflux, drhodx, bins=n_bins)
    density1[density1 == 0] = np.nan
    ax1.pcolormesh(bhflux_bins, drhodx_bins, density1.transpose() / n_samples, cmap='turbo', vmin=vmin, vmax=vmax)
    
    ax1.plot([0,0], [-2, 0], '-k', linewidth=0.7)
    ax1.set_xlabel('Buoyancy flux (10$^{-5}$ kg m$^{-2}$ s$^{-1}$)')
    ax1.set_ylabel(r'$\frac{\partial\rho}{\partial x}$ (10$^{-5}$ kg m$^{-3}$ m$^{-1}$)')
    ax1.set_xlim([-4, 2])
    ax1.set_ylim([-2, 0.0])
    add_subtitle(ax1, r'(a) Buoyancy vs $\frac{\partial\rho}{\partial x}$')
    r1, p1 = stats.pearsonr(bhflux, drhodx)
    ax1.text(-4, -0.25, f'  $R$={np.round(r1, 2)}, $p$<0.05', va='top')
    
    ax2 = plt.subplot(2, 2, 2)
    # ax2.scatter(df_analysis['sst_sh'].values - df_analysis['sst_dp'].values, df_dswt['drhodx'].values * 10**5, marker='x', s=20, c=color_neg)
    density2, dsst_bins, drhodx_bins = np.histogram2d(dsst, drhodx, bins=n_bins)
    density2[density2 == 0] = np.nan
    ax2.pcolormesh(dsst_bins, drhodx_bins, density2.transpose() / n_samples, cmap='turbo', vmin=vmin, vmax=vmax)
    
    ax2.set_xlabel('SST inshore - SST offshore ($^o$C)')
    ax2.set_yticklabels([])
    ax2.set_xlim([-6, 0])
    ax2.set_ylim([-2, 0.0])
    add_subtitle(ax2, r'(b) SST difference vs $\frac{\partial\rho}{\partial x}$')
    r2, p2 = stats.pearsonr(dsst, drhodx)
    ax2.text(-6, -0.25, f'  $R$={np.round(r2, 2)}, $p$<0.05', va='top')
    
    ax3 = plt.subplot(2, 2, 3)
    # ax3.scatter(df_dswt['drhodx'].values * 10**5, df_dswt['transport_50m'].values / (24*60*60), marker='x', s=20, c=color_neg)
    density3, drhodx_bins, transport_bins = np.histogram2d(drhodx, transport, bins=n_bins)
    density3[density3 == 0] = np.nan
    ax3.pcolormesh(drhodx_bins, transport_bins, density3.transpose() / n_samples, cmap='turbo', vmin=vmin, vmax=vmax)
    
    ax3.set_xlabel(r'$\frac{\partial\rho}{\partial x}$ (10$^{-5}$ kg m$^{-3}$ m$^{-1}$)')
    ax3.set_ylabel('Transport (m$^2$ s$^{-1}$)')
    ax3.set_xlim([-2, 0])
    ax3.set_ylim([0, 1.4])
    add_subtitle(ax3, r'(c) $\frac{\partial\rho}{\partial x}$ vs transport')
    
    ax4 = plt.subplot(2, 2, 4)
    density4, drhodx_bins, vel_bins = np.histogram2d(drhodx, vel, bins=n_bins)
    density4[density4 == 0] = np.nan
    c = ax4.pcolormesh(drhodx_bins, vel_bins, density4.transpose() / n_samples, cmap='turbo', vmin=vmin, vmax=vmax)
    ax4.set_xlabel(r'$\frac{\partial\rho}{\partial x}$ (10$^{-5}$ kg m$^{-3}$ m$^{-1}$)')
    ax4.set_ylabel('Velocity (m s$^{-1}$)')
    ax4.set_xlim([-2, 0])
    ax4.set_ylim([0, 0.13])
    add_subtitle(ax4, r'(d) $\frac{\partial\rho}{\partial x}$ vs velocity')
    
    # colorbar
    l2, b2, w2, h2 = ax2.get_position().bounds
    l4, b4, w4, h4 = ax4.get_position().bounds
    cax = fig.add_axes([l4 + w4 + 0.02, b4, 0.02, b2 + h2 - b4])
    cbar = plt.colorbar(c, cax=cax)
    cbar.set_label('Samples (fraction)')
    
    # --- theoretical estimates
    x = np.arange(-2.0, 0.01, 0.01) * 10**-5
    
    # note: for both the gravity current and Nof velocity estimates
    # I am using delta_rho ~ drho/dx delta_x
    # this is not strictly the delta_rho meant here.
    # instead, delta_rho should be the difference between
    # the dense plume and ambient water.
    
    # gravity current estimate
    # u = sqrt(g * delta_rho/rho0 * h)
    dx = np.sqrt(1/grid_ds.pm.values*1/grid_ds.pn.values)
    dx = np.nanmean(dx[grid_ds.h.values <= 100])
    h_dswt = np.nanmean(df_dswt.thickness.values)
    y_grav = np.sqrt(G / RHO0 * abs(x) * dx * h_dswt)
    ax4.plot(x*10**5, y_grav, '-w', linewidth=2)
    ax4.plot(x*10**5, y_grav, '-k', label='grav.')
    
    # geostrophic velocity estimate
    # u = 1/6 * h/f * g/rho0 * drho/dx
    # from Gawarkiewicz & Chapman (1995)
    h = 50
    y_geo = 1/6 * h/F * G/RHO0 * x
    ax4.plot(x*10**5, y_geo, '-w', linewidth=2)
    ax4.plot(x*10**5, y_geo, '--k', label='geo.')
    
    # legend
    ax4.legend(loc='upper right')
    
    # map inset panel b
    l2, b2, w2, h2 = ax2.get_position().bounds
    ax22 = fig.add_axes([l2+0.75*w2, b2-0.01, 0.2*w2, 0.4*h2], projection=ccrs.PlateCarree())
    plot_basic_map(ax22, lon_range_default, lat_range_default)
    deep_shallow = np.empty(grid_ds.h.shape)*np.nan
    l_shallow = grid_ds.h <= 20.
    l_deep = np.logical_and(grid_ds.h >= 50, grid_ds.h <= 300)
    deep_shallow[l_shallow] = 0
    deep_shallow[l_deep] = 1
    
    cmap = ListedColormap(['#9794be', '#d7789d'])
    ax22.pcolormesh(grid_ds.lon_rho.values, grid_ds.lat_rho.values, deep_shallow, vmin=0, vmax=1, cmap=cmap, zorder=1)
    
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()

def plot_dswt_per_wind_dir(df_dswt:pd.DataFrame, df_analysis:pd.DataFrame, output_path=None, show=False):
    df_analysis = convert_df_to_daily_means(df_analysis)
    l_southerly, l_northerly, l_onshore, l_offshore = _split_into_wind_dirs(df_analysis)
    
    def _plot_per_wind_dir(ax, l_wind_dir):
        df_wind_dir = df_analysis[l_wind_dir]
        df_dswt_wind_dir = df_dswt[l_wind_dir]
        
        time = np.array([pd.to_datetime(d) for d in df_dswt_wind_dir['time'].values])
        # l_summer = get_l_months(time, [1, 2, 3, 10, 11, 12])
        # l_time = get_l_months(time, [5, 6, 7])

        x = df_wind_dir['wind_vel'].values
        y = df_dswt_wind_dir['transport_50m'].values / (24*60*60)
        c = df_dswt_wind_dir['drhodx'].values * 10**5
        ax.set_ylabel('Transport (m$^2$ s$^{-1}$)')
        ax.set_xlabel(r'Wind speed (m s$^{-1}$)')
        ax.set_xlim([0, 17.5])
        ax.set_ylim([0, 1.4])
        
        # ax.scatter(x[l_summer], y[l_summer], marker='x', s=20, c=color_pos, label='Warm months')
        # ax.scatter(x[l_winter], y[l_winter], marker='x', s=20, c=color_neg, label='DSWT months')
        
        # cmap_subset_blues = plt.colormaps['Blues_r'](np.linspace(0.0, 0.8, 100))
        # new_cmap = ListedColormap(cmap_subset_blues, name='blues_subset')
        c = ax.scatter(x, y, marker='x', s=20, c=c, cmap='viridis', vmin=-2.0, vmax=0.0)
        return c

    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(hspace=0.1, wspace=0.12)
    
    ax1 = plt.subplot(2, 2, 1)
    _ = _plot_per_wind_dir(ax1, l_southerly)
    add_subtitle(ax1, '(a) Upwelling favorable (southerly)')
    ax1.set_xticklabels([])
    ax1.set_xlabel('')
    
    ax2 = plt.subplot(2, 2, 2)
    _ = _plot_per_wind_dir(ax2, l_northerly)
    add_subtitle(ax2, '(b) Downwelling favorable (northerly)')
    ax2.set_xticklabels([])
    ax2.set_xlabel('')
    ax2.set_yticklabels([])
    ax2.set_ylabel('')
    
    ax3 = plt.subplot(2, 2, 3)
    _ = _plot_per_wind_dir(ax3, l_onshore)
    add_subtitle(ax3, '(c) Onshore (westerly)')
    
    ax4 = plt.subplot(2, 2, 4)
    c = _plot_per_wind_dir(ax4, l_offshore)
    add_subtitle(ax4, '(d) Offshore (easterly)')
    ax4.set_yticklabels([])
    ax4.set_ylabel('')
    
    # ax4.legend(loc='upper right', bbox_to_anchor=(1.0, 0.9))
    l4, b4, w4, h4 = ax4.get_position().bounds
    l2, b2, w2, h2 = ax2.get_position().bounds
    cax = fig.add_axes([l4+w4+0.02, b4, 0.02, b2+h2-b4])
    cbar = plt.colorbar(c, cax=cax)
    cbar.set_label(r'$\frac{\partial\rho}{\partial x}$ (10$^{-5}$ kg m$^{-3}$ m$^{-1}$)')
    
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()

def plot_dswt_timeseries_evolution(df_dswt:pd.DataFrame, df_analysis:pd.DataFrame, highlight_dates=None, output_path=None, show=False):
    df_analysis = convert_df_to_daily_means(df_analysis)
    
    time = np.array([pd.to_datetime(d) for d in df_dswt['time'].values])
    months = [4, 5, 6, 7, 8, 9]
    l_time = get_l_months(time, months)
    
    xlim = [time[l_time][0], time[l_time][-1]]
    xticks = np.array([datetime(2017, m, 15) for m in months])
    xticklabels = np.array([d.strftime('%b') for d in xticks])
    
    fig = plt.figure(figsize=(8, 6))
    
    ax1 = plt.subplot(4, 1, (2, 4))
    
    # DSWT transport
    ax1.plot(time[l_time], df_dswt['transport_50m'].values[l_time]/(24*60*60), '-', color=color_transport, linewidth=2)
    ax1.set_ylabel('Transport (m$^2$ s$^{-1}$)')
    ax1.set_ylim([0, 1.2])
    plot_monthly_grid(ax1, 2017, alpha=0.7)
    ax1.set_xlim(xlim)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels)
    add_subtitle(ax1, '(b) DSWT transport and SST difference')
    
    ax11 = ax1.twinx()
    ax11.plot(time[l_time], df_analysis['sst_sh'].values[l_time]-df_analysis['sst_dp'].values[l_time], '-k', label='SST inshore - SST offshore ($^o$C)')
    ax11.set_ylim([-5, 0])
    ax11.set_ylabel('SST inshore - SST offshore ($^o$C)')
    
    color_y_axis(ax1, color_transport, 'left')
    
    # wind arrows
    wind_dir = df_analysis['wind_dir'].values[l_time]
    wind_vel = df_analysis['wind_vel'].values[l_time]
    wind_u = np.cos(np.deg2rad(wind_dir))
    wind_v = np.sin(np.deg2rad(wind_dir))

    l_disruptive = np.logical_and(wind_dir >= 45, wind_dir<=225)

    ax0 = plt.subplot(4, 1, 1)
    ax0.plot(time[l_time], wind_vel, '-', color='#808080', linewidth=0.5)
    ax0.quiver(time[l_time][~l_disruptive], wind_vel[~l_disruptive], wind_u[~l_disruptive], wind_v[~l_disruptive], wind_vel[~l_disruptive], cmap='Blues', angles='uv', scale=40, width=0.003)
    ax0.quiver(time[l_time][l_disruptive], wind_vel[l_disruptive], wind_u[l_disruptive], wind_v[l_disruptive], wind_vel[l_disruptive], cmap='Reds', angles='uv', scale=40, width=0.003)
    ax0.set_ylim([0, 18])
    plot_monthly_grid(ax0, 2017, alpha=0.7)
    ax0.set_xlim(xlim)
    ax0.set_xticklabels([])
    ax0.set_yticks(np.arange(0, 20, 5))
    ax0.set_ylabel('Wind speed\n(m s$^{-1}$)')
    add_subtitle(ax0, '(a) Wind')
    
    # fill disruptive area
    ax1.fill_between(time[l_time], 0,1, where=l_disruptive, transform=ax1.get_xaxis_transform(), color=color_pos, alpha=0.1, ec='None')
    
    if highlight_dates is not None:
        l_highlight = get_l_time_range(time[l_time], highlight_dates[0], highlight_dates[1])
        ax1.fill_between(time[l_time], 0, 1, where=l_highlight, transform=ax1.get_xaxis_transform(), facecolor='None', ec='k')
    
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()

def plot_dswt_event(df_transport:pd.DataFrame, dates:list[datetime],
                    df_analysis:WindTimeseries, transect_name='t240',
                    vmin=20.0, vmax=23.0, cmap='RdYlBu_r',
                    output_path=None, show=False):
    
    dates = np.array([pd.to_datetime(d) for d in dates])
    
    time = np.array([pd.to_datetime(t) for t in df_transport['time'].values])

    df_analysis = convert_df_to_daily_means(df_analysis)
    
    transects = read_transects_in_lon_lat_range_from_json('input/transects/cwa_transects.json',
                                                          [114.0, 116.0], [-33.0, -31.0])
    config = read_config('cwa')
    
    fig = plt.figure(figsize=(8, 10))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.rcParams['font.size'] = 10
    
    qscale = 2
    qwidth = 0.007
    
    def _plot_dswt_map(ax:plt.axes, map_date:datetime):
        map_start = map_date
        map_end = map_start + timedelta(days=1) - timedelta(hours=1)
        
        # load ROMS data
        model_input_dir = get_dir_from_json('cwa')
        input_file = [p for p in os.listdir(f'{model_input_dir}{map_start.year}/') if map_start.strftime("%Y%m%d") in p][0]
        input_path = f'{model_input_dir}{map_start.year}/{input_file}'
        grid_file = f'{model_input_dir}grid.nc'
        roms_ds = load_roms_data(input_path, grid_file=grid_file)
        
        lon = roms_ds.lon_rho.values
        lat = roms_ds.lat_rho.values
        h = roms_ds.h.values
        
        thin = 3
        lon_thin = roms_ds.lon_rho.values[::thin, ::thin]
        lat_thin = roms_ds.lat_rho.values[::thin, ::thin]
        u = np.nanmean(roms_ds['u_eastward'].values, axis=0)[0, :, :]
        v = np.nanmean(roms_ds['v_northward'].values, axis=0)[0, :, :]
        temp = np.nanmean(roms_ds['temp'].values, axis=0)[0, :, :]
        temp[h > 200] = np.nan
        
        # load dswt data
        l_time = get_l_time_range(time, map_start, map_end)
        transport_map = get_transport_map(df_transport, l_time, lon.shape)
        
        l_dswt = transport_map > 0
        l_dswt[h > 100] = False
        
        u[~l_dswt] = np.nan
        v[~l_dswt] = np.nan
        u[h > 100] = np.nan
        v[h > 100] = np.nan
        u_thin = u[::thin, ::thin]
        v_thin = v[::thin, ::thin]
        
        # wind data
        time_wind = np.array([pd.to_datetime(d) for d in df_analysis.time.values])
        l_wind = get_l_time_range(time_wind, map_start, map_end)
        wind_vel = np.nanmean(df_analysis['wind_vel'].values[l_wind])
        wind_dir = np.nanmean(df_analysis['wind_dir'].values[l_wind])
        
        # plots
        plot_basic_map(ax, lon_range_default, lat_range_default,
                       meridians_default, parallels_default, full_resolution=False)
        plot_contours(lon, lat, h,
                    lon_range_default, lat_range_default,
                    ax=ax, show=False, color='w',
                    clevels=[50, 200],
                    linewidths=[1.4, 1.4], clabel=False)
        plot_contours(lon, lat, h,
                    lon_range_default, lat_range_default,
                    ax=ax, show=False,
                    clevels=[50, 200],
                    linewidths=[0.8, 0.8], clabel=False)
        c = ax.pcolormesh(lon, lat, temp, vmin=vmin, vmax=vmax, cmap=cmap, zorder=1)
        q = ax.quiver(lon_thin, lat_thin, u_thin, v_thin, scale=qscale, width=qwidth, color='#252525')
        
        l, b, w, h = ax.get_position().bounds
        axw = fig.add_axes([l+0.05*w, b+0.4*h, w/10, h/10])
        axw.text(0, 0, f'{np.round(wind_vel, 0)} m/s', rotation=270-wind_dir, bbox=dict(boxstyle='rarrow', fc="#81a1ab", ec='k'), fontsize=8)
        axw.set_axis_off()
        
        return c, q, roms_ds
    
    def _plot_dswt_transect(ax:plt.axes, transect_name:str, roms_ds:xr.Dataset, ax_map=None):
        eta = transects[transect_name]['eta']
        xi = transects[transect_name]['xi']
        
        transect_ds = select_roms_transect_from_known_coordinates(roms_ds, eta, xi)
        
        (t_dswt, _, vel, thickness, _, distance, _, _, h, _, _, _, _) = determine_dswt_along_transect(transect_ds, config)
        
        # model transect
        x = transect_ds.distance.values
        z = transect_ds.z_rho.values
        
        x_dswt = distance
        z_dswt = -h + thickness
        u_dswt = vel
        
        if len(t_dswt) != 0:
            d = np.nanmean(transect_ds.temp.values[t_dswt, :, :], axis=0)
        else:
            d = np.nanmean(transect_ds.temp.values, axis=0)
        
        h = -transect_ds.h.values
        
        ax.pcolormesh(x, z, d, cmap=cmap, vmin=vmin, vmax=vmax)
        q = ax.quiver(x_dswt, z_dswt, u_dswt, np.zeros(len(u_dswt)), scale=qscale, width=qwidth, color='#252525')
        ax.fill_between(x, -110, h, color='#d2d2d2', edgecolor='k')
        ax.set_ylim([-100, 0])
        ax.set_ylabel('Depth (m)')
        ax.set_yticks([0, -25, -50, -75, -100])
        ax.set_yticklabels([0, 25, 50, 75, 100])
        ax.set_xlim([x[0], x[-1]])
        ax.set_xlabel('Distance along transect (m)')
        
        # add transect location to map
        if ax_map is not None:
            ax_map.plot(transect_ds.lon_rho.values, transect_ds.lat_rho.values, '-', color='w', linewidth=2)
            ax_map.plot(transect_ds.lon_rho.values, transect_ds.lat_rho.values, '-', color='#C70039', linewidth=1)
        
        return q
        
    # --- maps
    ax1 = plt.subplot(10, 3, (1, 10), projection=ccrs.PlateCarree())
    c, _, roms_ds1 = _plot_dswt_map(ax1, dates[0])
    
    ax2 = plt.subplot(10, 3, (2, 11), projection=ccrs.PlateCarree())
    _, _, roms_ds2 = _plot_dswt_map(ax2, dates[1])
    ax2.set_yticklabels([])
    
    ax3 = plt.subplot(10, 3, (3, 12), projection=ccrs.PlateCarree())
    _, _, roms_ds3 = _plot_dswt_map(ax3, dates[2])
    ax3.set_yticklabels([])
    
    # new row
    ax4 = plt.subplot(10, 3, (16, 25), projection=ccrs.PlateCarree())
    _, _, roms_ds4 = _plot_dswt_map(ax4, dates[3])
    
    ax5 = plt.subplot(10, 3, (17, 26), projection=ccrs.PlateCarree())
    _, _, roms_ds5 = _plot_dswt_map(ax5, dates[4])
    ax5.set_yticklabels([])
    
    ax6 = plt.subplot(10, 3, (18, 27), projection=ccrs.PlateCarree())
    _, _, roms_ds6 = _plot_dswt_map(ax6, dates[5])
    ax6.set_yticklabels([])
    
    # --- transects
    ax11 = plt.subplot(10, 3, 13)
    _ = _plot_dswt_transect(ax11, transect_name, roms_ds1, ax1)
    add_subtitle(ax11, f'(a) {(dates[0]).strftime("%d-%m-%Y")}', location='lower left')
    
    ax22 = plt.subplot(10, 3, 14)
    _ = _plot_dswt_transect(ax22, transect_name, roms_ds2, ax2)
    ax22.set_yticklabels([])
    ax22.set_ylabel('')
    add_subtitle(ax22, f'(b) {dates[1].strftime("%d-%m-%Y")}', location='lower left')
    
    ax33 = plt.subplot(10, 3, 15)
    _ = _plot_dswt_transect(ax33, transect_name, roms_ds3, ax3)
    ax33.set_yticklabels([])
    ax33.set_ylabel('')
    add_subtitle(ax33, f'(c) {dates[2].strftime("%d-%m-%Y")}', location='lower left')
    
    # new row
    ax44 = plt.subplot(10, 3, 28)
    _ = _plot_dswt_transect(ax44, transect_name, roms_ds4, ax4)
    add_subtitle(ax44, f'(d) {dates[3].strftime("%d-%m-%Y")}', location='lower left')
    
    ax55 = plt.subplot(10, 3, 29)
    _ = _plot_dswt_transect(ax55, transect_name, roms_ds5, ax5)
    ax55.set_yticklabels([])
    ax55.set_ylabel('')
    add_subtitle(ax55, f'(e) {dates[4].strftime("%d-%m-%Y")}', location='lower left')
    
    ax66 = plt.subplot(10, 3, 30)
    q = _plot_dswt_transect(ax66, transect_name, roms_ds6, ax6)
    ax66.set_yticklabels([])
    ax66.set_ylabel('')
    add_subtitle(ax66, f'(f) {dates[5].strftime("%d-%m-%Y")}', location='lower left')
    
    # move transect plots up
    l11, _, w11, h11 = ax11.get_position().bounds
    l22, _, w22, h22 = ax22.get_position().bounds
    l33, _, w33, h33 = ax33.get_position().bounds
    
    _, b1, _, _ = ax1.get_position().bounds
    bnew1 = b1 - h11 - 0.03
    ax11.set_position([l11, bnew1, w11, h11])
    ax22.set_position([l22, bnew1, w22, h22])
    ax33.set_position([l33, bnew1, w33, h33])
    
    # move second row down
    l4, _, w4, h4 = ax4.get_position().bounds
    l5, _, w5, h5 = ax5.get_position().bounds
    l6, _, w6, h6 = ax6.get_position().bounds
    
    _, b11, _, _ = ax11.get_position().bounds
    bnew = b11 - h4 - 0.08
    ax4.set_position([l4, bnew, w4, h4])
    ax5.set_position([l5, bnew, w5, h5])
    ax6.set_position([l6, bnew, w6, h6])
    
    # move second row transect plots
    l44, _, w44, h44 = ax44.get_position().bounds
    l55, _, w55, h55 = ax55.get_position().bounds
    l66, _, w66, h66 = ax66.get_position().bounds
    
    l4, b4, w4, h4 = ax4.get_position().bounds
    bnew2 = b4 - h44 - 0.03
    ax44.set_position([l44, bnew2, w44, h44])
    ax55.set_position([l55, bnew2, w55, h55])
    ax66.set_position([l66, bnew2, w66, h66])
    
    # colorbar
    l44, b44, w44, h44 = ax44.get_position().bounds
    l66, _, w66, h66 = ax66.get_position().bounds
    cax = fig.add_axes([l44, b44-0.1, l66 + w66 - l44, 0.02])
    cbar = plt.colorbar(c, cax=cax, orientation='horizontal')
    cbar.set_label('Temperature ($^o$C)')
    
    # quiver
    qkey = ax66.quiverkey(q, X=0.55, Y=-0.8, U=0.2, label='0.2 m s$^{-1}$', labelpos='E', transform=ax11.transAxes)
    
    if output_path is not None:
        plt.savefig(output_path, bbox_extra_artists=(qkey,), bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()

def plot_overall_export_comparison(ucross_ds:xr.Dataset, df_timeseries:pd.DataFrame,
                                   output_path=None, show=False):
    
    # monthly DSWT transport
    time = np.array([pd.to_datetime(d) for d in df_timeseries['time'].values])
    time_m, dswt_transport_m , dswt_transport_std = get_monthly_means(time, df_timeseries['transport_50m'].values)
    
    # monthly overall positive cross-shelf transport
    lon_range = [114.0, 116.0]
    lat_range = [-33.0, -31.0]
    grid_ds = xr.load_dataset(f'{get_dir_from_json("cwa")}grid.nc')
    grid_ds = select_roms_subset(grid_ds, time_range=None, lon_range=lon_range, lat_range=lat_range)
    
    z_w = get_z(grid_ds.Vtransform.values, grid_ds.s_w.values, grid_ds.h.values, grid_ds.Cs_w.values, grid_ds.hc.values)
    grid_ds.coords['z_w'] = (['s_w', 'eta_rho', 'xi_rho'], z_w)
    
    delta_z = np.diff(z_w, axis=0)
    ucross_ds['delta_z'] = (['s_rho', 'eta_rho', 'xi_rho'], delta_z)
    
    ocean_time = np.array([pd.to_datetime(d) for d in ucross_ds['ocean_time'].values])
    
    lon_contour50, lat_contour50, contour_length = get_roms_contour_coordinates(grid_ds, lon_range, lat_range, 50)
    ucross_ds_contour50 = get_roms_ds_along_contour(ucross_ds, grid_ds, lon_contour50, lat_contour50)
    ucross = ucross_ds_contour50['u_cross'].values
    ucross[ucross <= 0] = np.nan
    ucross_overall = np.nansum(
        np.nansum(
            ucross * ucross_ds_contour50['delta_z'].values, axis=1) * ucross_ds_contour50['dx'].values,
        axis=1) / contour_length
    
    time_m, ucross_m, ucross_std = get_monthly_means(ocean_time, ucross_overall)
    
    # figure
    xlim = [datetime(2017, 1, 1), datetime(2017, 12, 31)]
    time_m_str = [t.strftime('%b') for t in time_m]
    
    dswt_percentage = (dswt_transport_m/(24*60*60)) / ucross_m * 100
    
    fig = plt.figure(figsize=(8, 5))
    plt.rcParams['font.size'] = 14
    
    # timeseries
    ax = plt.axes()
    plot_monthly_histogram(time_m, dswt_percentage, ylabel='Offshore transport due to DSWT (%)',
                           ylim=[0, 50], time_is_center=True, color=color_transport,
                           ax=ax, show=False)
    ax.set_xlim(xlim)
    ax.set_xticks(time_m)
    ax.set_xticklabels(time_m_str)
    plot_monthly_grid(ax, 2017)
    add_subtitle(ax, 'Percentage cross-shelf transport related to DSWT')
    
    # save and show figure
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()

# --- Supplemental ---
def plot_yearly_events(dswt_events:DswtEvents, years:list, output_path=None, show=False):
    
    xlim = [datetime(years[0], 1, 1), datetime(years[-1], 12, 31)]
    
    def _get_yearly_event_means_stds(parameter:str) -> list[np.ndarray]:
        means = []
        stds = []
        
        for i in range(len(years)):
            values_y = np.array([getattr(d, parameter) for d in dswt_events.events[i]])
            means.append(np.nanmean(values_y))
            stds.append(np.nanstd(values_y))
            
        return np.array(means), np.array(stds)
    
    time = dswt_events.time
    duration, duration_std = _get_yearly_event_means_stds('duration')
    mean_vel, mean_vel_std = _get_yearly_event_means_stds('mean_vel')
    mean_thickness, mean_thickness_std = _get_yearly_event_means_stds('mean_thickness')
    mean_h, mean_h_std = _get_yearly_event_means_stds('mean_h')
    
    fig = plt.figure(figsize=(8, 12))
    plt.rcParams['font.size'] = 12
    
    # n events
    ylim1 = [0, 30]
    yticks1 = np.arange(0, 40, 10)
    ax1 = plt.subplot(5, 1, 1)
    plot_histogram_multiple_years(time, dswt_events.n_events, ylabel='Events (#)',
                                  ylim=ylim1, color=color_neg, ax=ax1)
    ax1.set_xlim(xlim)
    ax1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax1.set_yticks(yticks1)
    add_subtitle(ax1, '(a) DSWT events')
    
    # duration
    ylim2 = [0, 10]
    yticks2 = np.arange(0, 12, 2)
    ax2 = plt.subplot(5, 1, 2)
    plot_histogram_multiple_years(time, duration, yerr=duration_std,
                                  color=color_transport, err_color=color_transport_std,
                                  ylabel='Duration (days)', ylim=ylim2,
                                  ax=ax2, show=False)
    ax2.set_xlim(xlim)
    ax2.set_xticklabels([])
    ax2.set_yticks(yticks2)
    add_subtitle(ax2, '(b) DSWT duration')
    
    # vel
    ylim3 = [0, 0.14]
    yticks3 = np.arange(0, 0.14, 0.02)
    ax3 = plt.subplot(5, 1, 3)
    plot_histogram_multiple_years(time, mean_vel, yerr=mean_vel_std,
                                  color=color_transport, err_color=color_transport_std,
                                  ylabel='Velocity (m s$^{-1}$)', ylim=ylim3,
                                  ax=ax3, show=False)
    ax3.set_xlim(xlim)
    ax3.set_xticklabels([])
    ax3.set_yticks(yticks3)
    add_subtitle(ax3, '(c) DSWT event velocities')
    
    # thickness
    ylim4 = [0, 20]
    yticks4 = np.arange(0, 25, 5)
    ax4 = plt.subplot(5, 1, 4)
    plot_histogram_multiple_years(time, mean_thickness, yerr=mean_thickness_std,
                                  color=color_transport, err_color=color_transport_std,
                                  ylabel='Thickness (m)', ylim=ylim4,
                                  ax=ax4, show=False)
    ax4.set_xlim(xlim)
    ax4.set_xticklabels([])
    ax4.set_yticks(yticks4)
    add_subtitle(ax4, '(d) DSWT event thicknesses')
    
    # h
    ylim5 = [0, 70]
    yticks5 = np.arange(40, 120, 20)
    ax5 = plt.subplot(10, 1, (9, 10))
    plot_histogram_multiple_years(time, mean_h, yerr=mean_h_std,
                                  color=color_transport, err_color=color_transport_std,
                                  ylabel='Shelf depth (m)', ylim=ylim5,
                                  ax=ax5, show=False)
    ax5.set_xlim(xlim)
    ax5.set_yticks(yticks5)
    add_subtitle(ax5, '(e) DSWT event depths reached')
    
    # --- write overall stats to csv ---
    def _get_overall_mean_std(parameter):
        values = []
        for i in range(len(years)):
            values.append([getattr(d, parameter) for d in dswt_events.events[i]])

        values_flat = []
        for i in range(len(values)):
            for j in range(len(values[i])):
                values_flat.append(values[i][j])
        values_flat = np.array(values_flat)

        mean_value = np.nanmean(values_flat)
        std_value = np.nanstd(values_flat)
        return mean_value, std_value
    
    df = pd.DataFrame(data=np.empty((5, 3)), columns=['parameter', 'mean', 'std'])
    df['parameter'] = ['events', 'duration', 'mean_vel', 'mean_thickness', 'mean_h']
    df.iloc[0, 1] = np.nanmean(dswt_events.n_events)
    df.iloc[0, 2] = np.nanstd(dswt_events.n_events)
    df.iloc[1, 1], df.iloc[1, 2] = _get_overall_mean_std('duration')
    df.iloc[2, 1], df.iloc[2, 2] = _get_overall_mean_std('mean_vel')
    df.iloc[3, 1], df.iloc[3, 2] = _get_overall_mean_std('mean_thickness')
    df.iloc[4, 1], df.iloc[4, 2] = _get_overall_mean_std('mean_h')
    
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        df.to_csv(f'{os.path.splitext(output_path)[0]}.csv', index=False)
    if show == True:
        plt.show()
    else:
        plt.close()

def dswt_animation(df_transport:pd.DataFrame,
                   start_date:datetime,
                   end_date:datetime,
                   df_analysis:pd.DataFrame,
                   fps=1, dpi=100,
                   output_path=None):
    
    df_analysis = convert_df_to_daily_means(df_analysis)
    
    model_input_dir = get_dir_from_json('cwa')
    grid_file = f'{model_input_dir}grid.nc'
    
    grid_ds = xr.load_dataset(grid_file)
    lon = grid_ds.lon_rho.values
    lat = grid_ds.lat_rho.values
    h = grid_ds.h.values
    
    n_frames = (end_date - start_date).days + 1
    
    time = np.array([pd.to_datetime(t) for t in df_transport['time'].values])
    
    scale = 5
    thin = 4
    lon_roms = lon[::thin, ::thin]
    lat_roms = lat[::thin, ::thin]
    
    writer = animation.PillowWriter(fps=fps)

    # plot map
    plt.rcParams.update({'font.size' : 15})
    plt.rcParams.update({'font.family': 'arial'})
    plt.rcParams.update({'figure.dpi': dpi})
    fig = plt.figure(figsize=(8, 8))
    fig.tight_layout()
    ax = plt.axes(projection=ccrs.PlateCarree())
    plot_basic_map(ax, lon_range_default, lat_range_default, meridians_default, parallels_default, full_resolution=True)
    plot_contours(lon, lat, h,
                  lon_range_default, lat_range_default,
                  ax=ax, show=False, color='w',
                  clevels=[25, 50, 100, 200],
                  linewidths=[2.0, 2.0, 2.0, 2.0])
    plot_contours(lon, lat, h,
                  lon_range_default, lat_range_default,
                  ax=ax, show=False,
                  clevels=[25, 50, 100, 200],
                  linewidths=[1.0, 1.0, 1.0, 1.0])

    # animated fields
    field = ax.pcolormesh(lon, lat, np.zeros(lon.shape), cmap='viridis', vmin=0.0, vmax=0.6)
    cbar = plt.colorbar(field)
    cbar.set_label('Transport (m$^2$ s$^{-1}$)')
    
    quivers = ax.quiver(lon_roms, lat_roms, np.zeros(lon_roms.shape), np.zeros(lat_roms.shape), scale=scale, color='#252525')
    ax.quiverkey(quivers, X=0.83, Y=-0.08, U=0.2, label='0.2 m/s', labelpos='E')

    # animated text
    ttl = ax.text(0.5, 1.04, '', transform=ax.transAxes,
                ha='center', va='top',
                bbox=dict(facecolor='w', alpha=0.3, edgecolor='w', pad=2))
    ttl.set_animated(True)
    
    ll, bb, ww, hh = ax.get_position().bounds
    axw = fig.add_axes([ll+0.75*ww, bb+0.82*hh, ww/5, hh/5])
    axw.set_axis_off()
    arrowt = axw.text(0, 0, '', rotation=0, bbox=dict(boxstyle='rarrow', fc='#70bed5', ec='k'), fontsize=15)
    arrowt.set_animated(True)
    
    def animate(i):
        map_start = start_date + timedelta(days=i)
        map_end = map_start + timedelta(days=1) - timedelta(hours=1)
        l_time = get_l_time_range(time, map_start, map_end)
        transport_map = get_transport_map(df_transport, l_time, lon.shape)
        transport_map = transport_map/(24*60*60)
        
        field.set_array(transport_map.ravel())
    
        # load ROMS data
        input_file = [p for p in os.listdir(f'{model_input_dir}{map_start.year}/') if map_start.strftime("%Y%m%d") in p][0]
        input_path = f'{model_input_dir}{map_start.year}/{input_file}'
        roms_ds = load_roms_data(input_path, grid_file=grid_file)
        u = np.nanmean(roms_ds.u_eastward.values, axis=0)[0, ::thin, ::thin]
        v = np.nanmean(roms_ds.v_northward.values, axis=0)[0, ::thin, ::thin]
        
        quivers.set_UVC(u, v)
        
        # wind data
        time_wind = np.array([pd.to_datetime(d) for d in df_analysis.time.values])
        l_wind = get_l_time_range(time_wind, map_start, map_end)
        wind_vel = np.nanmean(df_analysis['wind_vel'].values[l_wind])
        wind_dir = np.nanmean(df_analysis['wind_dir'].values[l_wind])
        arrowt.set_text(f'{np.round(wind_vel, 0)} m/s')
        arrowt.set_rotation(270-wind_dir)

        title = map_start.strftime('%d %b %Y')
        ttl.set_text(title)

        return ttl

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=n_frames, blit=False)
    if output_path is not None:
        log.info(f'Saving animation to: {output_path}')
        anim.save(output_path, writer=writer)
    else:
        plt.show()

# --- Temporary / still to clean ---

def plot_transport_maps(df_transport:pd.DataFrame, start_date:datetime, end_date:datetime,
                        wind:WindTimeseries,
                        ylim_transport=[0, 0.6], n_cols=3, n_rows=3,
                        output_path=None, show=False):
    
    n_days = (end_date - start_date).days + 1
    time = np.array([pd.to_datetime(t) for t in df_transport['time'].values])
    
    alphabet = list(string.ascii_lowercase)
    
    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.rcParams['font.size'] = 12
    
    for i in range(n_days):
        ax = plt.subplot(n_rows, n_cols, i+1, projection=ccrs.PlateCarree())
        
        if np.mod(i, n_cols) == 0:
            ymarkers = 'left'
        else:
            ymarkers = 'off'
            
        if i >= n_cols*(n_rows-1):
            xmarkers = 'bottom'
        else:
            xmarkers = 'off'
        
        map_start = start_date + timedelta(days=i)
        map_end = map_start + timedelta(days=1) - timedelta(hours=1)
        
        # load ROMS data
        model_input_dir = get_dir_from_json('cwa')
        input_file = [p for p in os.listdir(f'{model_input_dir}{map_start.year}/') if map_start.strftime("%Y%m%d") in p][0]
        input_path = f'{model_input_dir}{map_start.year}/{input_file}'
        grid_file = f'{model_input_dir}grid.nc'
        roms_ds = load_roms_data(input_path, grid_file=grid_file)
        scale = 3
        thin = 3
        lon_roms = roms_ds.lon_rho.values[::thin, ::thin]
        lat_roms = roms_ds.lat_rho.values[::thin, ::thin]
        u = np.nanmean(roms_ds.u_eastward.values, axis=0)[0, ::thin, ::thin]
        v = np.nanmean(roms_ds.v_northward.values, axis=0)[0, ::thin, ::thin]
        
        lon = roms_ds.lon_rho.values
        lat = roms_ds.lat_rho.values
        h = roms_ds.h.values
        
        # load dswt data
        l_time = get_l_time_range(time, map_start, map_end)
        transport_map = get_transport_map(df_transport, l_time, lon.shape)
        
        # wind data
        l_wind = get_l_time_range(wind.time, map_start, map_end)
        wind_vel = np.nanmean(wind.vel[l_wind])
        wind_dir = np.nanmean(wind.dir[l_wind])
        
        # plots
        plot_basic_map(ax, lon_range_default, [-30.8, -33.0],
                       meridians_default, parallels_default, full_resolution=True,
                       xmarkers=xmarkers, ymarkers=ymarkers)
        plot_contours(lon, lat, h,
                    lon_range_default, lat_range_default,
                    ax=ax, show=False, color='w',
                    clevels=[25, 50, 100, 200],
                    linewidths=[2.0, 4.0, 2.0, 2.0], clabel=False)
        plot_contours(lon, lat, h,
                    lon_range_default, lat_range_default,
                    ax=ax, show=False,
                    clevels=[25, 50, 100, 200],
                    linewidths=[1.0, 2.0, 1.0, 1.0], clabel=False)
        c = ax.pcolormesh(lon, lat, transport_map/(24*60*60), vmin=ylim_transport[0], vmax=ylim_transport[1], cmap='viridis', zorder=1)
        q = ax.quiver(lon_roms, lat_roms, u, v, scale=scale, color='#252525')
        
        l, b, w, h = ax.get_position().bounds
        axw = fig.add_axes([l+0.74*w, b+0.81*h, w/5, h/5])
        axw.text(0, 0, f'{np.round(wind_vel, 0)} m/s', rotation=270-wind_dir, bbox=dict(boxstyle='rarrow', fc='#70bed5', ec='k'), fontsize=8)
        axw.set_axis_off()
        
        add_subtitle(ax, f'({alphabet[i]}) {map_start.strftime("%d %b %Y")}')
        
    # colorbar
    l, b, w, h = ax.get_position().bounds
    cax = fig.add_axes([l+w+0.02, b, 0.02, h*n_rows+0.03])
    cbar = plt.colorbar(c, cax=cax)
    cbar.set_label('DSWT cross-shelf transport (m$^2$ s$^{-1}$)')
    
    # quiver
    qkey = ax.quiverkey(q, X=0.55, Y=1.04, U=0.2, label='0.2 m s$^{-1}$', labelpos='E')
    
    if output_path is not None:
        plt.savefig(output_path, bbox_extra_artists=(qkey,), bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()

def plot_dswt_map(df_transport:pd.DataFrame,
                  grid_ds:xr.Dataset,
                  time_req=None,
                  variable='transport',
                  norm=(24*60*60),
                  cbar_label='Transport (m$^2$ s$^{-1}$)',
                  vmin=0, vmax=1.4,
                  output_path=None,
                  show=False):
    # DSWT map
    if time_req == None:
        l_time = np.ones(len(df_transport)).astype(bool) # mean map over all times
    else:
        time = np.array([pd.to_datetime(d) for d in df_transport['time'].values])
        l_time = get_l_time_range(time, time_req, time_req)
    
    transport_map = get_transport_map(df_transport, l_time, grid_ds.lon_rho.shape, variable=variable)
    
    l_mask = grid_ds.h.values > 100.
    transport_map[l_mask] = np.nan
    transport_map[transport_map == 0] = np.nan
    transport_map = transport_map / norm
    
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
    
    c = ax.pcolormesh(grid_ds.lon_rho.values, grid_ds.lat_rho.values, transport_map, cmap='viridis', vmin=vmin, vmax=vmax)
    
    # colorbar
    ll, bb, ww, hh = ax.get_position().bounds
    cax = fig.add_axes([ll+ww+0.02, bb, 0.02, hh])
    cbar = plt.colorbar(c, cax=cax)
    cbar.set_label(cbar_label)
    
    # save and show figure
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()

def plot_us_ue_comparison(df_analysis:pd.DataFrame, output_path=None, show=False):    
    
    time = np.array([pd.to_datetime(d) for d in df_analysis['time'].values])
    l_summer = np.logical_or(get_l_time_range(time, datetime(2017, 10, 1), datetime(2017, 12, 31)),
                             get_l_time_range(time, datetime(2017, 1, 1), datetime(2017, 3, 31)))
    l_winter = get_l_time_range(time, datetime(2017, 4, 1), datetime(2017, 9, 30))
    
    df_summer = df_analysis.loc[l_summer]
    df_winter = df_analysis.loc[l_winter]
    
    def _plot_northerly_southerly(ax, df):
        l_southerly, l_northerly, _, _ = _split_into_wind_dirs(df)
        df_southerly = df.loc[l_southerly]
        df_northerly = df.loc[l_northerly]
        
        x_southerly = df_southerly['wind_vel'].values
        l_use_s = x_southerly >= 2.5
        x_northerly = df_northerly['wind_vel'].values
        l_use_n = x_northerly >= 2.5
        y_southerly = (df_southerly['Uss'].values - df_southerly['Tes'].values) / df_southerly['Tes'].values
        y_northerly = (df_northerly['Uss'].values - df_northerly['Tes'].values) / df_northerly['Tes'].values
        
        ax.scatter(x_southerly[l_use_s], y_southerly[l_use_s], marker='o', s=20, c=color_neg, label='Upwelling favorable')
        ax.scatter(x_northerly[l_use_n], y_northerly[l_use_n], marker='o', s=20, c=color_pos, label='Downwelling favorable')
        ax.plot(xlim, [1, 1], '--k')
        ax.plot(xlim, [0, 0], '-k')
    
    xlim = [2, 15]
    
    fig = plt.figure(figsize=(8, 6))
    plt.subplots_adjust(wspace=0.3)
    
    # ax1 = plt.subplot(1, 2, 1)
    # ax1.scatter(x_southerly[l_use_s], df_southerly['hes'].values[l_use_s], marker='o', s=20, c='k', label='h$_{Ekman, s}$')
    # ax1.scatter(x_northerly[l_use_n], df_northerly['hes'].values[l_use_n], marker='o', s=20, c='k')
    # ax1.scatter(x_southerly[l_use_s], -df_southerly['zss'].values[l_use_s], marker='o', s=20, c=color_neg, label=r'$\delta_s$')
    # ax1.scatter(x_northerly[l_use_n], -df_northerly['zss'].values[l_use_n], marker='o', s=20, c=color_pos, label=r'$\delta_s$')
    # ax1.plot(xlim, [-50, -50], '--k')
    # ax1.set_ylabel('Layer depth (m)')
    # ax1.set_xlabel('Wind speed (m s$^{-1}$)')
    # ax1.set_xlim(xlim)
    # ax1.legend(loc='lower left')
    # ax1.set_ylim([-101, -10])
    # add_subtitle(ax1, '(a) Surface boundary layer')
    
    ax1 = plt.subplot(1, 2, 1)
    _plot_northerly_southerly(ax1, df_summer)
    ax1.legend(loc='lower right')
    ax1.set_ylabel('$U_{s}$ / $U_{E,s}$')
    ax1.set_xlim(xlim)
    ax1.set_ylim([-10, 20])
    add_subtitle(ax1, '(a) Surface cross-shelf transport (Oct-Mar)')
    
    ax2 = plt.subplot(1, 2, 2)
    _plot_northerly_southerly(ax2, df_winter)
    ax2.set_xlabel('Wind speed (m s$^{-1}$)')
    ax2.set_xlim(xlim)
    ax2.set_ylim([-10, 20])
    add_subtitle(ax2, '(b) Surface cross-shelf transport (Apr-Sep)')
    
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()

def plot_tests(df_dswt:pd.DataFrame, df_analysis:pd.DataFrame, output_path=None, show=False):
    df_analysis = convert_df_to_daily_means(df_analysis)
    
    fig = plt.figure(figsize=(8, 5))
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(df_analysis['bflux_sh'].values, df_dswt['drhodx'].values, marker='x', s=20, c=color_neg)
    ax1.set_xlabel('Bflux')
    ax1.set_ylabel('drhodx')
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(df_dswt['drhodx'].values, df_dswt['transport_50m'].values, marker='x', s=20, c=color_neg)
    ax2.set_xlabel('drhodx')
    ax2.set_ylabel('transport')
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(df_analysis['bflux_sh'].values, df_dswt['transport_50m'].values, marker='x', s=20, c=color_neg)
    ax3.set_xlabel('Bflux')
    ax3.set_ylabel('transport')
    
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(df_dswt['drhodx'].values, df_dswt['vel_50m'].values, marker='x', s=20, c=color_neg)
    ax4.set_xlabel('drhodx')
    ax4.set_ylabel('vel')
    
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(df_dswt['drhodx'].values, df_dswt['thickness_50m'].values, marker='x', s=20, c=color_neg)
    ax5.set_xlabel('drhodx')
    ax5.set_ylabel('thickness')
    
    
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()  

def plot_tests_per_wind_dir(df_dswt:pd.DataFrame, df_analysis:pd.DataFrame, output_path=None, show=False):
    df_analysis = convert_df_to_daily_means(df_analysis)
    l_southerly, l_northerly, l_onshore, l_offshore = _split_into_wind_dirs(df_analysis)
    
    def _plot_per_wind_dir(ax, l_wind_dir):
        df_wind_dir = df_analysis[l_wind_dir]
        df_dswt_wind_dir = df_dswt[l_wind_dir]
        
        time = np.array([pd.to_datetime(d) for d in df_dswt_wind_dir['time'].values])
        l_summer = get_l_months(time, [1, 2, 3, 10, 11, 12])
        l_winter = get_l_months(time, [4, 5, 6, 7, 8, 9])

        x = df_wind_dir['wind_vel'].values[l_winter]
        y = df_dswt_wind_dir['transport_50m'].values[l_winter] / (24*60*60)
        # ax.set_ylabel('Transport (m$^2$ s$^{-1}$)')
        # ax.set_xlabel(r'Wind speed (m s$^{-1}$)')
        # ax.set_xlim([0, 17.5])
        # ax.set_ylim([0, 1.4])
        
        # ax.scatter(x[l_summer], y[l_summer], marker='x', s=20, c=color_pos, label='Warm months')
        # ax.scatter(x[l_winter], y[l_winter], marker='x', s=20, c=color_neg, label='DSWT months')
        bins = np.arange(0, 20, 2)
        n_wind, _ = np.histogram(x, bins=bins)
        n_dswt = []
        for i in range(len(bins)-1):
            l_bin = np.logical_and(x >= bins[i], x < bins[i+1])
            n_dswt.append(np.sum(y[l_bin] != 0))
            center_bin = bins[i] + 1
        n_dswt = np.array(n_dswt)
            
        ax.bar(bins[:-1] + 0.5*np.diff(bins), n_wind, width=-0.8, align='edge')
        ax.bar(bins[:-1] + 0.5*np.diff(bins), n_dswt, width=0.8, align='edge')
        

    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(hspace=0.1, wspace=0.12)
    
    ax1 = plt.subplot(2, 2, 1)
    _plot_per_wind_dir(ax1, l_southerly)
    add_subtitle(ax1, '(a) Upwelling favorable (southerly)')
    ax1.set_xticklabels([])
    ax1.set_xlabel('')
    
    ax2 = plt.subplot(2, 2, 2)
    _plot_per_wind_dir(ax2, l_northerly)
    add_subtitle(ax2, '(b) Downwelling favorable (northerly)')
    ax2.set_xticklabels([])
    ax2.set_xlabel('')
    ax2.set_yticklabels([])
    ax2.set_ylabel('')
    
    ax3 = plt.subplot(2, 2, 3)
    _plot_per_wind_dir(ax3, l_onshore)
    add_subtitle(ax3, '(c) Onshore (westerly)')
    
    ax4 = plt.subplot(2, 2, 4)
    _plot_per_wind_dir(ax4, l_offshore)
    add_subtitle(ax4, '(d) Offshore (easterly)')
    ax4.set_yticklabels([])
    ax4.set_ylabel('')
    
    # ax4.legend(loc='upper right', bbox_to_anchor=(1.0, 0.9))
    
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()    

def plot_overall_lvc_map(ucross_ds:xr.Dataset, times='all',
                         cmap='viridis', vmin=0, vmax=0.1,
                         output_path=None, show=False):
    ocean_time = np.array([pd.to_datetime(d) for d in ucross_ds.ocean_time.values])
    l_jan = get_l_time_range(ocean_time, datetime(2017, 1, 1), datetime(2017, 1, 31))
    l_jul = get_l_time_range(ocean_time, datetime(2017, 6, 1), datetime(2017, 6, 30))
    
    if times == 'all':
        h_3d = np.repeat(np.expand_dims(ucross_ds.h.values, axis=0), len(ucross_ds.ocean_time), axis=0)
        lvc_values = 1/h_3d * ucross_ds['L_vc'].values # remove 1/h when new file ready
        ubar_values = ucross_ds['u_bar'].values
    elif times == 'lb':
        lvc_values = ucross_ds['L_vc_lb'].values
        ubar_values = ucross_ds['u_bar_lb'].values
    elif times == 'sb':
        lvc_values = ucross_ds['L_vc_sb'].values
        ubar_values = ucross_ds['u_bar_sb'].values
    else:
        raise ValueError(f'Unknown option for times: {times}. Valid options are: "all", "lb", or "sb".')
    
    lvc_values_scaled = lvc_values#/ubar_values
    lvc_values_scaled = np.nanmean(lvc_values_scaled, axis=0) # time mean
    lvc_values_scaled[lvc_values_scaled == 0] = np.nan
    
    lvc_values_jan = np.nanmean(lvc_values[l_jan, :, :], axis=0)
    lvc_values_jan[lvc_values_jan == 0] = np.nan
    lvc_values_jul = np.nanmean(lvc_values[l_jul, :, :], axis=0)
    lvc_values_jul[lvc_values_jul == 0] = np.nan
    
    fig = plt.figure(figsize=(8, 5))
    
    ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
    plot_basic_map(ax1, lon_range_default, lat_range_default,
                   meridians=meridians_default, parallels=parallels_default)
    # c = ax.pcolormesh(ucross_ds.lon_rho, ucross_ds.lat_rho, lvc_values_scaled, cmap=cmap, vmin=vmin, vmax=vmax)
    c = ax1.pcolormesh(ucross_ds.lon_rho, ucross_ds.lat_rho, lvc_values_jan, cmap=cmap, vmin=vmin, vmax=vmax)
    plot_contours(ucross_ds.lon_rho.values, ucross_ds.lat_rho.values, ucross_ds.h.values,
                  lon_range_default, lat_range_default,
                  ax=ax1, show=False,
                  clevels=[25, 50, 100, 200],
                  linewidths=[1.0, 2.0, 1.0, 1.0])
    
    ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
    plot_basic_map(ax2, lon_range_default, lat_range_default,
                   meridians=meridians_default, parallels=parallels_default)
    # c = ax.pcolormesh(ucross_ds.lon_rho, ucross_ds.lat_rho, lvc_values_scaled, cmap=cmap, vmin=vmin, vmax=vmax)
    c = ax2.pcolormesh(ucross_ds.lon_rho, ucross_ds.lat_rho, lvc_values_jul, cmap=cmap, vmin=vmin, vmax=vmax)
    plot_contours(ucross_ds.lon_rho.values, ucross_ds.lat_rho.values, ucross_ds.h.values,
                  lon_range_default, lat_range_default,
                  ax=ax2, show=False,
                  clevels=[25, 50, 100, 200],
                  linewidths=[1.0, 2.0, 1.0, 1.0])
    ax2.set_yticklabels([])
    
    # colorbar
    l, b, w, h = ax2.get_position().bounds
    cax = fig.add_axes([l+w+0.1, b, 0.02, h])
    cbar = plt.colorbar(c, cax=cax)
    cbar.set_label('Measure of vertical complexity')
    
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    input_dir = f'{get_dir_from_json("output")}'
    input_dir_processed = f'{get_dir_from_json("output")}processed/'
    plot_dir = f'{get_dir_from_json("plots")}cwa/'
    input_dir_analysis = get_dir_from_json("analysis")
    grid_file = f'{get_dir_from_json("cwa")}grid.nc'
    
    years = np.arange(2000, 2024)
    
    # ---------------------------------------------------
    # Introduction
    # ---------------------------------------------------
    glider_data = GliderData.read_from_netcdf(f'{get_dir_from_json("glider_data")}IMOS_ANFOG_BCEOPSTUV_20160512T034541Z_SL502_FV01_timeseries_END-20160530T020243Z.nc')
    glider_data.get_data_in_time_frame(datetime(2016, 5, 13, 8, 0), datetime(2016, 5, 15, 8, 0))
    glider_ds = convert_glider_data_to_transect_data(glider_data)
    bathy_ds = xr.load_dataset(get_dir_from_json("bathy")) # GA bathymetry data
    global_dswt_df = pd.read_csv(f'{get_dir_from_json("wcs")}global_dswt_locations.csv')
    plot_overview_map(glider_ds, bathy_ds, global_dswt_df, output_path=f'{plot_dir}overview.jpg')
    
    # ---------------------------------------------------
    # Methods
    # ---------------------------------------------------
    # glider versus model
    
    
    # transects + performance
    
    
    # ---------------------------------------------------
    # Results
    # ---------------------------------------------------
    
    # --- WCS data ---
    ucross_ds = xr.load_dataset(f'{input_dir_analysis}cross-shelf/ucross_2017.nc')
    df_dswt_2017 = pd.read_csv(f'{input_dir_processed}dswt_timeseries_2017.csv')
    df_analysis_2017 = pd.read_csv(f'{input_dir_analysis}analysis_2017.csv')
    df_transport_2017 = pd.read_csv(f'{input_dir_processed}dswt_transport_2017.csv')
    df_transport_2017_nointerp = pd.read_csv(f'{input_dir_processed}dswt_transport_2017_no-interp.csv')
    
    grid_ds = xr.load_dataset(grid_file)
    
    df_analysis = read_df_from_multiple_csvs(input_dir_analysis, years, 'analysis_')
    df_dswt = read_df_from_multiple_csvs(input_dir_processed, years, 'dswt_timeseries_')
    df_transport = read_df_from_multiple_csvs(input_dir_processed, years, 'dswt_transport_')
    
    dswt_events = DswtEvents.calculate_from_df_timeseries(df_dswt, years, req_months=[5, 6, 7])
    dswt_events_2017 = DswtEvents.calculate_from_df_timeseries(df_dswt_2017, [2017], req_months=[5, 6, 7])
    
    # --- WCS general cross-shelf transport ---
    plot_u_bar_overview(ucross_ds, output_path=f'{plot_dir}ubar_overview.jpg')
    plot_u_bar_seasonality_maps(ucross_ds, output_path=f'{plot_dir}ubar_seasonality.jpg')
    plot_u_prime_evolution(ucross_ds, df_analysis_2017, output_path=f'{plot_dir}uprime_timeseries.jpg')
    
    plot_us_ub_dynamics(df_analysis_2017, output_path=f'{plot_dir}Us_Ub_Ue_comparison.jpg')
    
    # --- WCS DSWT ----
    plot_overall_transport_map_with_monthly_climatology(df_dswt, df_analysis, df_transport, grid_ds, output_path=f'{plot_dir}dswt_climatology_map.jpg')
    
    plot_dswt_forcing(df_dswt, df_analysis, grid_ds, output_path=f'{plot_dir}dswt_forcing.jpg')
    plot_dswt_per_wind_dir(df_dswt, df_analysis, output_path=f'{plot_dir}dswt_wind_dir.jpg')
    
    event_start = datetime(2017, 6, 17)
    event_end = datetime(2017, 6, 22)
    plot_dswt_timeseries_evolution(df_dswt_2017, df_analysis_2017, highlight_dates=[event_start, event_end],
                                   output_path=f'{plot_dir}dswt_timeseries_evolution.jpg')

    plot_yearly_events(dswt_events, years, output_path=f'{plot_dir}dswt_event_statistics.jpg')
    plot_dswt_event(df_transport_2017_nointerp, np.arange(event_start, event_end + timedelta(days=1), timedelta(days=1)),
                    df_analysis_2017, output_path=f'{plot_dir}dswt_event_example.jpg')
    
    # --- WCS cross-shelf export comparison with DSWT ---
    plot_overall_export_comparison(ucross_ds, df_dswt_2017, output_path=f'{plot_dir}dswt_export_contribution.jpg')
    
    
    # ---------------------------------------------------
    # SI
    # ---------------------------------------------------
    # plot_yearly_events(dswt_events, years, output_path=f'{plot_dir}dswt_events_statistics.jpg')
    
    # ---------------------------------------------------
    # Temp
    # ---------------------------------------------------
    # plot_tests(df_dswt, df_analysis, output_path=f'{plot_dir}temp.jpg')
    # plot_tests_per_wind_dir(df_dswt, df_analysis, output_path=f'{plot_dir}temp.jpg')
    
    # plot_us_ue_comparison(pd.read_csv(f'{input_dir_analysis}analysis_2017.csv'), output_path=f'{plot_dir}Us_Ue_comparison.jpg')
    
    # start_date = datetime(2017, 5, 1)
    # end_date = datetime(2017, 5, 31)
    # dswt_animation(df_transport_2017, start_date, end_date, df_analysis_2017, output_path=f'{plot_dir}cwa_dswt_May2017.gif')
    