from plot_tools.basic_timeseries import plot_histogram_multiple_years, plot_monthly_histogram
from read_climate_indices import read_dmi_data, read_mei_data, read_yearly_fremantle_msl, read_monthly_fremantle_msl
from surface_fluxes import read_surface_fluxes_from_csvs
from read_meteo_data import read_wind_from_csvs
from tools.timeseries import get_yearly_means, get_monthly_means, add_month_to_time, get_l_time_range
from tools import log
from tools.files import get_dir_from_json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

plot_interannual_variation = False
plot_specific_year = True

means = 'monthly'
specific_years = [2015, 2019, 2022]

years = np.arange(2000, 2023)
model = 'cwa'
lon_range = [114.0, 116.0]
lat_range = [-33.0, -31.0]
input_dir = 'output/'

plot_dir = get_dir_from_json('plots')
output_path_interannual = f'{plot_dir}dswt_climate_indices_{means}.jpg'
output_str_specific_years = f'{plot_dir}dswt_'
show = False

ocean_blue = '#25419e'
color_pos = '#900C3F'
color_neg = '#1e1677'

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
# domain string
lon_range_str = f'{int(np.floor(lon_range[0]))}-{int(np.ceil(lon_range[1]))}'
lon_range_unit = 'E' if lon_range[0] > 0 else 'W'
lat_range_str = f'{int(abs(np.floor(lat_range[0])))}-{int(abs(np.ceil(lat_range[1])))}'
lat_range_unit = 'S' if lat_range[0] < 0 else 'S'
domain = f'{lon_range_str}{lon_range_unit}_{lat_range_str}{lat_range_unit}'

# --- load DSWT data ---

def read_f_dswt_from_csvs(input_paths:list[str], means='monthly') -> tuple[np.ndarray[datetime], np.ndarray[float]]:
    time = np.array([])
    f_dswt = np.array([])
    for input_path in input_paths:
        df = pd.read_csv(input_path)
        time_daily = pd.to_datetime(df['time'].values)
        f_dswt_daily = df['f_dswt'].values
        
        if means == 'yearly':
            time_y, f_dswt_y = get_yearly_means(time_daily, f_dswt_daily)
        elif means == 'monthly':
            time_y, f_dswt_y = get_monthly_means(time_daily, f_dswt_daily)
        elif means == 'daily':
            time_y = time_daily
            f_dswt_y = f_dswt_daily
        else:
            log.info(f'Unknown mean method requested: {means}. Using daily values.')
            time_y = time_daily
            f_dswt_y = f_dswt_daily
        
        time = np.concatenate((time, time_y))
        f_dswt = np.concatenate((f_dswt, f_dswt_y))
        
    return time, f_dswt

input_paths = []
for year in years:
    input_paths.append(f'{input_dir}{model}_{year}_{domain}.csv')

time, f_dswt = read_f_dswt_from_csvs(input_paths, means=means)

# --- load climate indices ---
time_dmi, dmi = read_dmi_data(year_range=[years[0], years[-1]])
if means == 'yearly':
    time_dmi, dmi = get_yearly_means(time_dmi, dmi)

time_mei, mei = read_mei_data(year_range=[years[0], years[-1]])
if means == 'yearly':
    time_mei, mei = get_yearly_means(time_mei, mei)
    
if means == 'yearly':
    time_fmsl, fmsl = read_yearly_fremantle_msl()
elif means == 'monthly':
    time_fmsl, fmsl = read_monthly_fremantle_msl()
l_time_fmsl = get_l_time_range(time_fmsl, time[0], time[-1].replace(year=time[-1].year+1))
time_fmsl = time_fmsl[l_time_fmsl]
fmsl = fmsl[l_time_fmsl]
fmsl = fmsl-np.nanmean(fmsl) # mean sea level anomaly

# --- load atmosphere data ---
input_paths_sflux = []
for year in years:
    input_paths_sflux.append(f'{input_dir}sflux/sflux_{model}_{year}_{domain}.csv')
time_sflux_d, shflux, ssflux = read_surface_fluxes_from_csvs(input_paths_sflux)

if means == 'yearly':
    time_sflux, shflux = get_yearly_means(time_sflux_d, shflux)
    _, ssflux = get_yearly_means(time_sflux_d, ssflux)
elif means == 'monthly':
    time_sflux, shflux = get_monthly_means(time_sflux_d, shflux)
    _, ssflux = get_monthly_means(time_sflux_d, ssflux)
    
input_paths_wind = []
for year in years:
    input_paths_wind.append(f'{input_dir}wind/wind_{model}_{year}_{domain}.csv')
time_wind_d, u, v, vel, dir = read_wind_from_csvs(input_paths_wind)

if means == 'yearly':
    time_wind, vel = get_yearly_means(time_wind_d, vel)
    _, dir = get_yearly_means(time_wind_d, dir)
elif means == 'monthly':
    time_wind, vel = get_monthly_means(time_wind_d, vel)
    _, dir = get_monthly_means(time_wind_d, dir)

# ---------------------------------------------------------
# Plot tools
# ---------------------------------------------------------
if means == 'yearly':
    ylim_dswt = [0, 40]
    ylim_shflux = [-100, 0]
    ylim_ssflux = [0, 1.5*10**(-6)]
    ylim_wind = [0, 7.5]
    ylim_fmsl = [-120, 120]
else:
    ylim_dswt = [0, 100]
    ylim_shflux = [-300, 300]
    ylim_ssflux = [-3*10**(-6), 3*10**(-6)]
    ylim_wind = [0, 10]
    ylim_fmsl = [-270, 270]

def _plot_yearly_grid(ax:plt.axes, years:list) -> plt.axes:
    ax.set_xticks([datetime(y, 7, 2) for y in years]) # ticks in the middle of the year
    plt.tick_params(axis='x', length=0)
    ax.set_xticklabels(years, rotation='vertical')
    
    ylim = ax.get_ylim()
    for y in years: # plot grid to show years
        ax.plot([datetime(y, 1, 1), datetime(y, 1, 1)], ylim, '-', color='#808080', alpha=0.2)
        
    return ax

def plot_monthly_grid(ax:plt.axes, year:int) -> plt.axes:
    plt.tick_params(axis='x', length=0)
    ylim = ax.get_ylim()
    for m in range(1, 13):
        date = datetime(year, m, 1)
        ax.plot([date, date], ylim, '-', color='#808080', alpha=0.2)
        
    return ax

def _color_y_axis(ax:plt.axes, color:str, spine_location:str):
    ax.spines[spine_location].set_color(color)
    ax.tick_params(axis='y', colors=color)
    ax.yaxis.label.set_color(color)
    return ax

def _wind_dir_ticks(ax:plt.axes) -> plt.axes:
    yticks = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    ytick_labels = ['N', '', 'E', '', 'S', '', 'W', '', 'N']
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    return ax

# ---------------------------------------------------------
# Interannual variation plots
# ---------------------------------------------------------
if plot_interannual_variation == True:
    xlim = [datetime(years[0], 1, 1), datetime(years[-1], 12, 31)]
    
    fig = plt.figure(figsize=(10, 12))

    # DSWT
    ax1 = plt.subplot(5, 1, 1)
    ax1 = plot_histogram_multiple_years(time, f_dswt*100,
                                        ylabel='DSWT occurrence (%)', ylim=ylim_dswt, color=ocean_blue,
                                        ax=ax1, show=False)
    ax1.set_xlim(xlim)
    ax1.set_xticklabels([])
    
    mean_f_dswt = np.nanmean(f_dswt)*100
    ax1.plot(xlim, [mean_f_dswt, mean_f_dswt], '--', color='#808080')

    # IOD
    ax2 = plt.subplot(5, 1, 2)
    ax2 = plot_histogram_multiple_years(time_dmi, dmi, ylabel='Dipole Mode Index ($^o$C)', ylim=[-2., 2.],
                                        color=[color_neg, color_pos], c_change=0,
                                        ax=ax2, show=False)
    ax2.plot(xlim, [0, 0], '-k')
    ax2.text(add_month_to_time(xlim[0], 3), 2.0, 'IOD+', rotation='vertical', va='top', ha='left')
    ax2.text(add_month_to_time(xlim[0], 3), -2.0, 'IOD-', rotation='vertical', va='bottom', ha='left')
    ax2.set_xlim(xlim)
    ax2.set_xticklabels([])

    # ENSO
    ax3 = plt.subplot(5, 1, 3)
    ax3 = plot_histogram_multiple_years(time_mei, mei, ylabel='Multivariate ENSO Index v2', ylim=[-3., 3.],
                                        color=[color_neg, color_pos], c_change=0,
                                        ax=ax3, show=False)
    ax3.plot(xlim, [0, 0], '-k')
    ax3.text(add_month_to_time(xlim[0], 3), 3.0, 'EL NINO', rotation='vertical', va='top', ha='left')
    ax3.text(add_month_to_time(xlim[0], 3), -3.0, 'LA NINA', rotation='vertical', va='bottom', ha='left')
    ax3.set_xlim(xlim)
    ax3.set_xticklabels([])
    
    ax33 = ax3.twinx()
    ax33.plot(time_fmsl, fmsl, '-', color=ocean_blue, linewidth=0.5)
    ax33.set_ylabel('Fremantle MSL anomaly\n(mm)')
    ax33 = _color_y_axis(ax33, ocean_blue, 'right')
    ax33.set_ylim(ylim_fmsl)

    # Surface fluxes
    ax4 = plt.subplot(5, 1, 4)
    ax4.plot(xlim, [0, 0], '-k')
    ax4.plot(time_sflux, shflux, '-', color=color_pos)
    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim_shflux)
    ax4.set_ylabel('Surface heat flux\n(W/m$^2$)')
    
    ax4 = _color_y_axis(ax4, color_pos, 'left')
    ax4 = _plot_yearly_grid(ax4, years)
    ax4.set_xticklabels([])
    
    mean_shflux = np.nanmean(shflux)
    ax4.plot(xlim, [mean_shflux, mean_shflux], '--', color=color_pos, alpha=0.5)
    
    ax5 = ax4.twinx()
    ax5.plot(time_sflux, ssflux, '-', linewidth=0.5, color=color_neg)
    ax5.set_ylabel('Surface salt flux (m/s)')
    ax5.set_ylim(ylim_ssflux)
    
    mean_ssflux = np.nanmean(ssflux)
    ax5.plot(xlim, [mean_ssflux, mean_ssflux], '--', color=color_neg, linewidth=0.5, alpha=0.5)
    
    ax5 = _color_y_axis(ax5, color_neg, 'right')

    # Wind
    ax6 = plt.subplot(5, 1, 5)
    ax6.plot(time_wind, vel, '-', color=ocean_blue)
    ax6.set_xlim(xlim)
    ax6.set_ylim(ylim_wind)
    ax6.set_ylabel('Wind speed (m/s)')
    
    mean_vel = np.nanmean(vel)
    ax6.plot(xlim, [mean_vel, mean_vel], '--', color=ocean_blue, alpha=0.5)
    
    ax6 = _color_y_axis(ax6, ocean_blue, 'left')
    ax6 = _plot_yearly_grid(ax6, years)
    
    ax7 = ax6.twinx()
    ax7.plot(time_wind, dir, '-', color='k', linewidth=0.5)
    ax7.set_ylim([0, 360])
    ax7.set_ylabel('Wind direction')
    ax7 = _wind_dir_ticks(ax7)
    
    mean_dir = np.nanmean(dir)
    ax7.plot(xlim, [mean_dir, mean_dir], '--', color='k', linewidth=0.5, alpha=0.5)
    
    ax7 = _color_y_axis(ax7, 'k', 'right')

    # Save
    if output_path_interannual is not None:
        # save figure
        plt.savefig(output_path_interannual, bbox_inches='tight', dpi=300)
    
    if show is True:
        plt.show()
    else:
        plt.close()

# ---------------------------------------------------------
# Specific year plot
# ---------------------------------------------------------
if plot_specific_year == True:
    for specific_year in specific_years:
        output_path_year = f'{output_str_specific_years}{specific_year}.jpg'
        xlim = [datetime(specific_year, 1, 1), datetime(specific_year, 12, 31)]
        
        fig = plt.figure(figsize=(10, 12))

        # DSWT
        ax1 = plt.subplot(5, 1, 1)
        l_time = get_l_time_range(time, xlim[0], xlim[1])
        ax1, xticks, xticklabels = plot_monthly_histogram(time[l_time], f_dswt[l_time]*100, ylabel='DSWT occurrence (%)',
                                                        ylim=ylim_dswt, color=ocean_blue, time_is_center=True,
                                                        ax=ax1, show=False)
        ax1 = plot_monthly_grid(ax1, specific_year)
        ax1.set_xlim(xlim)
        ax1.set_xticklabels([])
        
        ax1.set_title(specific_year)

        # IOD
        ax2 = plt.subplot(5, 1, 2)
        l_time_dmi = get_l_time_range(time_dmi, xlim[0], xlim[1])
        ax2, _, _ = plot_monthly_histogram(time_dmi[l_time_dmi], dmi[l_time_dmi], ylabel='Dipole Mode Index ($^o$C)', ylim=[-2., 2.],
                                        color=[color_neg, color_pos], c_change=0, time_is_center=True,
                                        ax=ax2, show=False)
        ax2 = plot_monthly_grid(ax2, specific_year)
        ax2.plot(xlim, [0, 0], '-k')
        ax2.text(xlim[1]+timedelta(days=3), 2.0, 'IOD+', rotation='vertical', va='top', ha='left')
        ax2.text(xlim[1]+timedelta(days=3), -2.0, 'IOD-', rotation='vertical', va='bottom', ha='left')
        ax2.set_xlim(xlim)
        ax2.set_xticklabels([])

        # ENSO
        ax3 = plt.subplot(5, 1, 3)
        l_time_mei = get_l_time_range(time_mei, xlim[0], xlim[1])
        ax3, _, _ = plot_monthly_histogram(time_mei[l_time_mei], mei[l_time_mei], ylabel='Multivariate ENSO Index v2', ylim=[-3., 3.],
                                        color=[color_neg, color_pos], c_change=0, time_is_center=True,
                                        ax=ax3, show=False)
        ax3 = plot_monthly_grid(ax3, specific_year)
        ax3.plot(xlim, [0, 0], '-k')
        ax3.text(xlim[1]+timedelta(days=3), 3.0, 'EL NINO', rotation='vertical', va='top', ha='left')
        ax3.text(xlim[1]+timedelta(days=3), -3.0, 'LA NINA', rotation='vertical', va='bottom', ha='left')
        ax3.set_xlim(xlim)
        ax3.set_xticklabels([])

        # Surface fluxes
        ax4 = plt.subplot(5, 1, 4)
        ax4.plot(xlim, [0, 0], '-k')
        ax4.plot(time_sflux, shflux, '-', color=color_pos)
        
        ax4.set_xticks(xticks)
        ax4.set_xticklabels([])
        ax4.set_xlim(xlim)
        ax4.set_ylim(ylim_shflux)
        ax4.set_ylabel('Surface heat flux\n(W/m$^2$)')
        
        ax4 = plot_monthly_grid(ax4, specific_year)
        ax4 = _color_y_axis(ax4, color_pos, 'left')
        
        ax5 = ax4.twinx()
        ax5.plot(time_sflux, ssflux, '-', linewidth=0.5, color=color_neg)
        ax5.set_xlim(xlim)
        ax5.set_ylabel('Surface salt flux (m/s)')
        ax5.set_ylim(ylim_ssflux)
        
        ax5 = _color_y_axis(ax5, color_neg, 'right')

        # Wind
        ax6 = plt.subplot(5, 1, 5)
        ax6.plot(time_wind, vel, '-', color=ocean_blue)
        
        ax6.set_xticks(xticks)
        ax6.set_xticklabels(xticklabels)
        ax6.set_xlim(xlim)
        ax6.set_ylim(ylim_wind)
        ax6.set_ylabel('Wind speed (m/s)')
        
        ax6 = plot_monthly_grid(ax6, specific_year)
        ax6 = _color_y_axis(ax6, ocean_blue, 'left')
        
        ax7 = ax6.twinx()
        ax7.plot(time_wind, dir, '-', color='k', linewidth=0.5)
        ax7.set_xlim(xlim)
        ax7.set_ylim([0, 360])
        ax7.set_ylabel('Wind direction')
        ax7 = _wind_dir_ticks(ax7)
        
        ax7 = _color_y_axis(ax7, 'k', 'right')

        # Save
        if output_path_year is not None:
            # save figure
            plt.savefig(output_path_year, bbox_inches='tight', dpi=300)
        
        if show is True:
            plt.show()
        else:
            plt.close()