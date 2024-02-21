from plot_tools.basic_timeseries import plot_histogram_multiple_years
from read_climate_indices import read_dmi_data, read_mei_data
from tools.timeseries import get_yearly_means, get_monthly_means, add_month_to_time
from tools import log
from tools.files import get_dir_from_json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ocean_blue = '#25419e'
color_pos = '#900C3F'
color_neg = '#1e1677'

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

def plot_histogram_dswt_climate_indices(xlim:list[float], output_path=None, show=True):

    fig = plt.figure(figsize=(10, 8))

    # DSWT
    ax1 = plt.subplot(3, 1, 1)
    ax1 = plot_histogram_multiple_years(time, f_dswt*100,
                                        ylabel='DSWT occurrence (%)', ylim=[0, 100], color=ocean_blue,
                                        ax=ax1, show=False)
    ax1.set_xlim(xlim)
    ax1.set_xticklabels([])

    # IOD
    ax2 = plt.subplot(3, 1, 2)
    ax2 = plot_histogram_multiple_years(time_dmi, dmi, ylabel='Dipole Mode Index ($^o$C)', ylim=[-2., 2.],
                                        color=[color_neg, color_pos], c_change=0,
                                        ax=ax2, show=False)
    ax2.text(add_month_to_time(xlim[1], 3), 2.0, 'Positive IOD', rotation='vertical', va='top', ha='left')
    ax2.text(add_month_to_time(xlim[1], 3), -2.0, 'Negative IOD', rotation='vertical', va='bottom', ha='left')
    ax2.set_xlim(xlim)
    ax2.set_xticklabels([])

    ax2.text(datetime(xlim[1].year-4, 7, 2), -1.8, f'R = {np.round(r_dmi, 2)}, p = {np.round(p_dmi, 3)}', va='bottom', ha='left')

    # ENSO
    ax3 = plt.subplot(3, 1, 3)
    ax3 = plot_histogram_multiple_years(time_mei, mei, ylabel='Multivariate ENSO Index v2', ylim=[-3., 3.],
                                        color=[color_neg, color_pos], c_change=0,
                                        ax=ax3, show=False)
    ax3.text(add_month_to_time(xlim[1], 3), 3.0, 'EL NINO', rotation='vertical', va='top', ha='left')
    ax3.text(add_month_to_time(xlim[1], 3), -3.0, 'LA NINA', rotation='vertical', va='bottom', ha='left')
    ax3.set_xlim(xlim)

    ax3.text(datetime(xlim[1].year-4, 7, 2), -2.8, f'R = {np.round(r_mei, 2)}, p = {np.round(p_mei, 3)}', va='bottom', ha='left')

    if output_path is not None:
        # save figure
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    if show is True:
        plt.show()
    else:
        plt.close()

if __name__ == '__main__':
    # ---------------------------------------------------------
    # USER INPUT
    # ---------------------------------------------------------
    
    years = np.arange(2000, 2022)
    model = 'cwa'

    lon_range = [114.0, 116.0]
    lat_range = [-33.0, -31.0]

    input_dir = 'output/'
    
    means = 'yearly'
    
    plot_dir = get_dir_from_json('plots')

    # ---------------------------------------------------------
    # file strings for DSWT files
    lon_range_str = f'{int(np.floor(lon_range[0]))}-{int(np.ceil(lon_range[1]))}'
    lon_range_unit = 'E' if lon_range[0] > 0 else 'W'
    lat_range_str = f'{int(abs(np.floor(lat_range[0])))}-{int(abs(np.ceil(lat_range[1])))}'
    lat_range_unit = 'S' if lat_range[0] < 0 else 'S'
    domain = f'{lon_range_str}{lon_range_unit}_{lat_range_str}{lat_range_unit}'
    
    # plotting xlim
    xlim = [datetime(years[0], 1, 1), datetime(years[-1], 12, 31)]
    
    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    
    # --- load DSWT data ---
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

    r_dmi, p_dmi = pearsonr(dmi, f_dswt)
    r_mei, p_mei = pearsonr(mei, f_dswt)

    # --- load atmosphere data ---
    # still need to do
    
    # ---------------------------------------------------------
    # Plots
    # ---------------------------------------------------------
    # --- histogram climate indices ---
    plot_histogram_dswt_climate_indices(xlim, output_path=f'{plot_dir}dswt_climate_indices_{means}.jpg', show=False)
