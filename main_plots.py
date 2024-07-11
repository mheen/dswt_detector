from tools.dswt_output import get_domain_str, get_monthly_dswt_values, get_yearly_dswt_values, read_multifile_timeseries, read_multifile_transport_maps
from tools.dswt_output import get_sflux_data, get_wind_data, get_monthly_atmosphere_data, get_yearly_atmosphere_data
from tools.dswt_output import get_monthly_yearly_mei_data
from tools.files import get_dir_from_json
from tools.timeseries import get_l_time_range
from transects import read_transects_in_lon_lat_range_from_json
from tools.config import read_config

from plot_tools.basic_timeseries import plot_histogram_multiple_years, plot_monthly_histogram
from plot_tools.basic_timeseries import plot_yearly_grid, plot_monthly_grid
from plot_tools.general import color_y_axis, add_subtitle, add_wind_dir_ticks
from plot_tools.basic_maps import plot_basic_map, plot_contours
from tools.timeseries import add_month_to_time, get_l_time_range

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from datetime import datetime, timedelta
import xarray as xr

# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------
plot_interannual_variation = False
plot_specific_years = False
plot_maps_specific_years = True

years = np.arange(2000, 2023)

model = 'cwa'
lon_range = [114.0, 116.0]
lat_range = [-33.0, -31.0]
meridians = [114.0, 115.0, 116.0]
parallels = [-33.0, -32.0, -31.0]

color_dswt = '#25419e'
color_transport = '#0e6e22'
color_pos = '#900C3F'
color_neg = '#1e1677'

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

domain = get_domain_str(lon_range, lat_range)
main_input_dir = f'output/{model}_{domain}/'

transects = read_transects_in_lon_lat_range_from_json(f'input/transects/{model}_transects.json', lon_range, lat_range)

config = read_config(model)

ds_grid = xr.load_dataset(f'{get_dir_from_json(model)}grid.nc')

# --- DSWT data ---
time, f_dswt, vel_dswt, transport_dswt = read_multifile_timeseries(main_input_dir, years)
time_m, f_dswt_m, vel_dswt_m, transport_dswt_m = get_monthly_dswt_values(time, f_dswt, vel_dswt, transport_dswt)
time_y, f_dswt_y, vel_dswt_y, transport_dswt_y = get_yearly_dswt_values(time, f_dswt, vel_dswt, transport_dswt)

# --- Climate index data ---
_, mei_m, _, mei_y = get_monthly_yearly_mei_data(years)

# --- Atmosphere data ---
_, shflux, ssflux = get_sflux_data(f'{main_input_dir}sflux/', years)
_, shflux_m, ssflux_m = get_monthly_atmosphere_data(time, shflux, ssflux)
_, shflux_y, ssflux_y = get_yearly_atmosphere_data(time, shflux, ssflux)

_, _, _, wind_vel, wind_dir = get_wind_data(f'{main_input_dir}wind/', years)
_, wind_vel_m, wind_dir_m = get_monthly_atmosphere_data(time, wind_vel, wind_dir)
_, wind_vel_y, wind_dir_y = get_yearly_atmosphere_data(time, wind_vel, wind_dir)

# ---------------------------------------------------------
# Plots
# ---------------------------------------------------------
if plot_interannual_variation == True:
    xlim = [datetime(years[0], 1, 1), datetime(years[-1], 12, 31)]
    
    # --- DSWT ---
    fig = plt.figure(figsize=(8, 6))
    # DSWT occurrence
    ax1 = plt.subplot(2, 1, 1)
    ax1 = plot_histogram_multiple_years(time_y, f_dswt_y*100,
                                        ylabel='DSWT occurrence (%)',
                                        ylim=[0, 40], color=color_dswt,
                                        ax=ax1, show=False)
    ax1 = plot_yearly_grid(ax1, years)
    ax1.set_xlim(xlim)
    ax1.set_xticklabels([])
    ax1 = add_subtitle(ax1, '(a) DSWT yearly mean occurrence')
    
    # DSWT transport
    ax2 = plt.subplot(2, 1, 2)
    ax2 = plot_histogram_multiple_years(time_y, transport_dswt_y*10**-6,
                                        ylabel='DSWT transport (10$^9$ m$^2$)',
                                        ylim=[0, 3.0],
                                        color=color_transport,
                                        ax=ax2, show=False)
    ax2 = plot_yearly_grid(ax2, years)
    ax2.set_xlim(xlim)
    ax2 = add_subtitle(ax2, '(b) DSWT yearly transport')
    
    plt.savefig('plots/dswt_interannual.jpg', bbox_inches='tight', dpi=300)
    plt.close()
    
    # --- DSWT and climate variability ---
    fig = plt.figure(figsize=(8, 10))
    # DSWT occurrence anomaly
    ax1 = plt.subplot(5, 1, 1)
    ax1 = plot_histogram_multiple_years(time_y, (f_dswt_y-np.nanmean(f_dswt_y))*100,
                                        ylabel='DSWT occurence\nanomaly (%)',
                                        ylim=[-10., 10.],
                                        color=color_dswt,
                                        ax=ax1, show=False)
    ax1.plot(xlim, [0, 0], '-k')
    ax1 = plot_yearly_grid(ax1, years)
    ax1.set_xlim(xlim)
    ax1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax1 = add_subtitle(ax1, '(a) DSWT yearly occurrence anomaly')
    
    # DSWT transport anomaly
    ax2 = plt.subplot(5, 1, 2)
    ax2 = plot_histogram_multiple_years(time_y, (transport_dswt_y-np.nanmean(transport_dswt_y))*10**-6,
                                        ylabel='DSWT transport\nanomaly (m$^3$/m)',
                                        ylim = [-1.5, 1.5],
                                        color=color_transport,
                                        ax=ax2, show=False)
    ax2.plot(xlim, [0, 0], '-k')
    ax2 = plot_yearly_grid(ax2, years)
    ax2.set_xlim(xlim)
    ax2.set_xticklabels([])
    ax2 = add_subtitle(ax2, '(b) DSWT yearly transport anomaly')
    
    # ENSO
    ax3 = plt.subplot(5, 1, 3)
    ax3 = plot_histogram_multiple_years(time_y, mei_y,
                                        ylabel='Multivariate ENSO\nIndex v2',
                                        ylim=[-2.0, 2.0],
                                        color=[color_neg, color_pos], c_change=0.0,
                                        ax=ax3, show=False)
    ax3.plot(xlim, [0, 0], '-k')
    ax3.text(add_month_to_time(xlim[1], 3), 2.0, 'EL NINO', rotation='vertical', va='top', ha='left')
    ax3.text(add_month_to_time(xlim[1], 3), -2.0, 'LA NINA', rotation='vertical', va='bottom', ha='left')
    ax3 = plot_yearly_grid(ax3, years)
    ax3.set_xlim(xlim)
    ax3.set_xticklabels([])
    ax3 = add_subtitle(ax3, '(c) El Nino indicator (MEI-v2)')
    
    # surface heat flux anomaly
    ax4 = plt.subplot(5, 1, 4)
    ax4.plot(time_y, shflux_y-np.nanmean(shflux_y), '-', color=color_pos)
    ax4.set_ylabel('Surface heat flux\nanomaly (W/m$^2$)')
    ax4.plot(xlim, [0, 0], '-k')
    ax4 = plot_yearly_grid(ax4, years)
    ax4.set_xlim(xlim)
    ax4.set_xticklabels([])
    ax4.set_ylim([-30., 30.])
    ax4 = color_y_axis(ax4, color_pos, 'left')
    
    # surface salt flux
    ax5 = ax4.twinx()
    ax5.plot(time_y, ssflux_y-np.nanmean(ssflux_y), '-', color=color_neg)
    ax5.set_ylabel('Surface salt flux\nanomaly (m/s)')
    ax5.set_ylim([-6.0*10**-7, 6.0*10**-7])
    ax5 = color_y_axis(ax5, color_neg, 'right')
    
    ax5 = add_subtitle(ax5, '(d) Surface flux anomalies')
    
    # wind speed & direction
    ax6 = plt.subplot(5, 1, 5)
    ax6.plot(time_y, wind_vel_y-np.nanmean(wind_vel_y), '-', color=color_dswt)
    ax6.plot(xlim, [0, 0], '-k')
    ax6.set_ylabel('Wind speed\nanomaly (m/s)')
    ax6.set_ylim([-0.4, 0.4])
    ax6 = plot_yearly_grid(ax6, years)
    ax6.set_xlim(xlim)
    ax6 = add_subtitle(ax6, '(f) Wind anomaly')

    plt.savefig('plots/dswt_interannual_climate.jpg', bbox_inches='tight', dpi=300)
    plt.close()
    
def get_monthly_climatology(time:np.ndarray, values:np.ndarray):
    values_mean = []
    for m in range(1, 13):
        l_month = [t.month == m for t in time]
        values_mean.append(np.nanmean(values[l_month]))
    
    return np.array(values_mean)

if plot_specific_years == True:
    f_dswt_mc = get_monthly_climatology(time_m, f_dswt_m)
    transport_dswt_mc = get_monthly_climatology(time_m, transport_dswt_m)
    shflux_mc = get_monthly_climatology(time_m, shflux_m)
    ssflux_mc = get_monthly_climatology(time_m, ssflux_m)
    wind_vel_mc = get_monthly_climatology(time_m, wind_vel_m)
    wind_dir_mc = get_monthly_climatology(time_m, wind_dir_m)
    
    for year in years:
        l_time = get_l_time_range(time_m, datetime(year, 1, 1), datetime(year, 12, 31))
        xlim = [datetime(year, 1, 1), datetime(year, 12, 31)]
        
        # --- Timeseries ---
        fig = plt.figure(figsize=(8, 10))
        # DSWT occurrence
        ax1 = plt.subplot(5, 1, 1)
        ax1, xticks, xlabels = plot_monthly_histogram(time_m[l_time], f_dswt_m[l_time]*100,
                                     ylabel='DSWT occurrence (%)',
                                     ylim=[0, 100],
                                     time_is_center=True,
                                     color=color_dswt,
                                     ax=ax1, show=False)
        ax1.plot(time_m[l_time], f_dswt_mc*100, '--', color=color_dswt)
        ax1 = plot_monthly_grid(ax1, year)
        ax1.set_xlim(xlim)
        ax1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax1 = add_subtitle(ax1, '(a) DSWT occurrence')
        ax1.set_title(year)
        
        # DSWT transport anomaly
        ax2 = plt.subplot(5, 1, 2)
        ax2, _, _ = plot_monthly_histogram(time_m[l_time], transport_dswt_m[l_time]*10**-6,
                                     ylabel='DSWT transport (m$^3$/m)',
                                     ylim=[0.0, 1.3],
                                     color=color_transport,
                                     time_is_center=True,
                                     ax=ax2, show=False)
        ax2.plot(time_m[l_time], transport_dswt_mc*10**-6, '--', color=color_transport)
        ax2 = plot_monthly_grid(ax2, year)
        ax2.set_xlim(xlim)
        ax2.set_xticklabels([])
        ax2 = add_subtitle(ax2, '(b) DSWT transport')
        
        # ENSO
        ax3 = plt.subplot(5, 1, 3)
        ax3, _, _ = plot_monthly_histogram(time_m[l_time], mei_m[l_time],
                                     ylabel='Multivariate ENSO\nIndex v2',
                                     ylim=[-2.5, 2.5],
                                     color=[color_neg, color_pos], c_change=0.0,
                                     time_is_center=True,
                                     ax=ax3, show=False)
        ax3.plot(xlim, [0, 0], '-k')
        ax3.text(xlim[1]+timedelta(days=3), 2.5, 'EL NINO', rotation='vertical', va='top', ha='left')
        ax3.text(xlim[1]+timedelta(3), -2.5, 'LA NINA', rotation='vertical', va='bottom', ha='left')
        ax3 = plot_monthly_grid(ax3, year)
        ax3.set_xlim(xlim)
        ax3.set_xticklabels([])
        ax3 = add_subtitle(ax3, '(c) El Nino indicator (MEI-v2)')
        
        # surface heat flux anomaly
        ax4 = plt.subplot(5, 1, 4)
        ax4.plot(time_m[l_time], shflux_m[l_time], '-', color=color_pos)
        ax4.plot(time_m[l_time], shflux_mc, '--', color=color_pos, linewidth=0.5)
        ax4.set_ylabel('Surface heat flux (W/m$^2$)')
        ax4.plot(xlim, [0, 0], '-k')
        ax4.set_xlim(xlim)
        ax4.set_xticklabels([])
        ax4.set_ylim([-300., 300.])
        ax4 = plot_monthly_grid(ax4, year)
        ax4 = color_y_axis(ax4, color_pos, 'left')
        
        # surface salt flux
        ax5 = ax4.twinx()
        ax5.plot(time_m[l_time], ssflux_m[l_time], '-', color=color_neg)
        ax5.plot(time_m[l_time], ssflux_mc, '--', color=color_neg, linewidth=0.5)
        ax5.set_ylabel('Surface salt flux (m/s)')
        ax5.set_ylim([-3.0*10**-6, 3.0*10**-6])
        ax5 = color_y_axis(ax5, color_neg, 'right')
        
        ax5 = add_subtitle(ax5, '(d) Surface fluxes')
        
        # wind speed & direction
        ax6 = plt.subplot(5, 1, 5)
        ax6.plot(time_m[l_time], wind_vel_m[l_time], '-', color=color_dswt)
        ax6.plot(time_m[l_time], wind_vel_mc, '--', color=color_dswt, linewidth=0.5)
        ax6.set_ylabel('Wind speed (m/s)')
        ax6.set_ylim([4.0, 10.0])
        ax6 = plot_monthly_grid(ax6, year)
        ax6.set_xticks(xticks)
        ax6.set_xticklabels(xlabels)
        ax6.set_xlim(xlim)
        ax6 = color_y_axis(ax6, color_dswt, 'left')
        ax6 = add_subtitle(ax6, '(f) Wind')
        
        ax7 = ax6.twinx()
        ax7.plot(time_m[l_time], wind_dir_m[l_time], '-', color='k')
        ax7.plot(time_m[l_time], wind_dir_mc, '--', color='k', linewidth=0.5)
        ax7.set_ylabel(['Wind direction'])
        ax7 = add_wind_dir_ticks(ax7)
        ax7.set_ylim([90, 270])

        plt.savefig(f'plots/dswt_{year}.jpg', bbox_inches='tight', dpi=300)
        plt.close()

if plot_maps_specific_years == True:
    model_input_dir = get_dir_from_json('cwa')
    grid_file = f'{model_input_dir}grid.nc'
    grid_ds = xr.open_dataset(grid_file)
    
    time, lon, lat, transport_dswt = read_multifile_transport_maps(main_input_dir, years,
                                                                   grid_ds.lon_rho.values,
                                                                   grid_ds.lat_rho.values)
    
    for year in years:
        
        l_time = get_l_time_range(time, datetime(year, 1, 1), datetime(year, 12, 31))
        transport_yearly = np.nanmean(transport_dswt[l_time, :, :], axis=0)
        
        fig = plt.figure(figsize=(6, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, lon_range, lat_range, meridians, parallels)
        ax = plot_contours(ds_grid.lon_rho.values, ds_grid.lat_rho.values, ds_grid.h.values,
                           lon_range=lon_range, lat_range=lat_range,
                           clevels=[10, 50, 100, 200],
                           ax=ax, show=False)
        
        c = ax.pcolormesh(lon, lat, transport_yearly, cmap='RdYlBu_r', vmin=5000, vmax=-5000)
        
        l, b, w, h = ax.get_position().bounds
        cbax = fig.add_axes([l+w+0.02, b, 0.03, h])
        cbar = plt.colorbar(c, cax=cbax)
        cbar.set_label('DSWT transport (m$^2$)')
        ax = add_subtitle(ax, f'Yearly mean DSWT transport', location='upper left')
        ax.set_title(year)
        
        plt.savefig(f'plots/dswt_map_{year}.jpg', bbox_inches='tight', dpi=300)
        plt.close()
    