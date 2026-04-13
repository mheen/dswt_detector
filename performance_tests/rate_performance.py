import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from performance_tests.plot_transects_for_manual_check import transects_plot
from readers.read_ocean_data import select_input_files, load_roms_data, select_roms_transect_from_known_coordinates
from transects import read_transects_in_lon_lat_range_from_json
from tools.files import get_dir_from_json
from plot_tools.general import add_subtitle
from plot_tools.basic_timeseries import plot_monthly_histogram
from tools.timeseries import get_monthly_sums, add_month_to_time
from tools import log
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import xarray as xr

def plot_performance_summary(df:pd.DataFrame, output_path=None, show=False):

    manual_dswt = df['manual_dswt'].values
    algorithm_dswt = df['algorithm_dswt'].values
    
    l_no_dswt = manual_dswt == 0
    l_dswt = manual_dswt == 1
    l_uncertain = manual_dswt == 0.5
    
    l_correct_no_dswt = algorithm_dswt[l_no_dswt] == 0
    n_correct_no_dswt = np.sum(l_correct_no_dswt)
    n_incorrect_no_dswt = np.sum(l_no_dswt) - n_correct_no_dswt
    transport_incorrect_no_dswt = df['transport'].values[l_no_dswt][~l_correct_no_dswt]
    transport_incorrect_no_dswt_sum = np.nansum(transport_incorrect_no_dswt)
    
    l_correct_dswt = algorithm_dswt[l_dswt] == 1
    n_correct_dswt = np.sum(l_correct_dswt)
    n_incorrect_dswt = np.sum(l_dswt) - n_correct_dswt
    transport_correct_dswt = df['transport'].values[l_dswt][l_correct_dswt]
    transport_correct_dswt_sum = np.nansum(transport_correct_dswt)
    # transport incorrectly DSWT is zero (so not needed)
    
    l_uncertain_no_dswt = algorithm_dswt[l_uncertain] == 0
    l_uncertain_dswt = algorithm_dswt[l_uncertain] == 1
    n_uncertain = np.sum(l_uncertain)
    transport_uncertain = df['transport'].values[l_uncertain]
    transport_uncertain = transport_uncertain[~np.isnan(transport_uncertain)]
    transport_uncertain_sum = np.nansum(transport_uncertain)
    
    # --- Summary plot
    fig = plt.figure(figsize=(10, 5))
    plt.subplots_adjust(wspace=0.5)
    
    # number of detections
    ax1 = plt.subplot(1, 2, 1)
    ax1.bar([1], [n_correct_no_dswt], color='#900C3F', label='Algorithm: no DSWT')
    ax1.bar([1], [n_incorrect_no_dswt], bottom=[n_correct_no_dswt], color='#25419e', label='Algorithm: DSWT')
    ax1.bar([2], [n_correct_dswt], color='#25419e')
    ax1.bar([2], [n_incorrect_dswt], bottom=[n_correct_dswt], color='#900C3F')
    ax1.bar([3], [np.sum(l_uncertain_dswt)], color='#25419e')
    ax1.bar([3], [np.sum(l_uncertain_no_dswt)], bottom=[np.sum(l_uncertain_dswt)], color='#900C3F')
    
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(['No DSWT', 'DSWT', 'Possible DSWT'])
    ax1.set_xlabel('Manual determination')
    
    ylim1 = np.ceil(max([np.sum(l_no_dswt), np.sum(l_dswt), np.sum(l_uncertain)]) / 50) * 50
    ax1.set_ylim([0, ylim1])
    ax1.set_ylabel('Tests (#)')
    
    ax11 = ax1.twinx()
    ax11.set_ylim([0, ylim1/len(manual_dswt) * 100])
    yticks = ax1.get_yticks()
    ax11.set_yticks(yticks/len(manual_dswt) * 100)
    ax11.set_ylabel('Tests (%)')
    
    ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 0.94))
    
    add_subtitle(ax1, '(a) Performance test outcomes')
    
    y_scale = 10**4
    # effect on transport
    ax2 = plt.subplot(1, 2, 2)
    bplot = ax2.boxplot([transport_incorrect_no_dswt / y_scale, transport_correct_dswt / y_scale, transport_uncertain / y_scale],
                        patch_artist=True, tick_labels=['No DSWT', 'DSWT', 'Possible DSWT'])
    for patch in bplot['boxes']:
        patch.set_facecolor('w')
    for median in bplot['medians']:
        median.set_color('k')
    
    ax2.set_xlabel('Manual determination')
    ax2.set_ylabel('Events transport (10$^4$ m$^2$ s$^{-1}$)')
    
    max_transport = np.array([max(transport_incorrect_no_dswt), max(transport_correct_dswt), max(transport_uncertain)]) / y_scale
    ylim2 = np.ceil(max(max_transport))
    ax2.set_ylim([0, ylim2])
    
    ax3 = ax2.twinx()
    ax3.scatter([1, 2, 3], np.array([transport_incorrect_no_dswt_sum, transport_correct_dswt_sum, transport_uncertain_sum]) / y_scale,
                c='#0e6e22', s=20, marker='x')
    ylim3 = np.ceil(max([transport_incorrect_no_dswt_sum / y_scale, transport_correct_dswt_sum / y_scale, transport_uncertain_sum / y_scale]) / 50) * 50
    ax3.set_ylim([0, ylim3])
    ax3.set_ylabel('Total transport (10$^4$ m$^2$ s$^{-1}$)')
    
    ax3.spines['right'].set_color('#0e6e22')
    ax3.tick_params(axis='y', colors='#0e6e22')
    ax3.yaxis.label.set_color('#0e6e22')
    
    add_subtitle(ax2, '(b) Effect on transport')
    
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        log.info(f'Saved performance summary to: {output_path}')
        
        data = np.array([[n_correct_no_dswt, n_incorrect_no_dswt, n_correct_dswt, n_incorrect_dswt, n_uncertain],
                         [0, transport_incorrect_no_dswt_sum, transport_correct_dswt_sum, 0, transport_uncertain_sum]])
        df_out = pd.DataFrame(data=data, columns=['No DSWT correct', 'No DSWT incorrect', 'DSWT correct', 'DSWT incorrect', 'uncertain'],
                              index=['n', 'transport (m2/s)'])
        output_csv = f'{os.path.splitext(output_path)[0]}.csv'
        df_out.to_csv(output_csv)
        
    if show == True:
        plt.show()
    else:
        plt.close()
    
def plot_monthly_performance(df:pd.DataFrame, output_path=None, show=False):

    time = np.array([datetime.strptime(str(d), '%Y%m%d%H%M') for d in df['time'].values])
    i_sort = np.argsort(time)
    time = time[i_sort]
    time_m, counts_m = get_monthly_sums(time, np.ones(len(time)))
    
    manual_dswt = df['manual_dswt'].values[i_sort]
    algorithm_dswt = df['algorithm_dswt'].values[i_sort]
    transport = df['transport'].values[i_sort]
    
    l_correct = manual_dswt == algorithm_dswt
    transport_correct = np.copy(transport)
    transport_correct[~l_correct] = 0.0
    
    l_incorrect1 = np.logical_and(manual_dswt == 1, algorithm_dswt == 0)
    l_incorrect2 = np.logical_and(manual_dswt == 0, algorithm_dswt == 1)
    l_incorrect = np.logical_or(l_incorrect1, l_incorrect2)
    transport_incorrect = np.copy(transport)
    transport_incorrect[~l_incorrect] = 0.0
    
    l_uncertain = manual_dswt == 0.5
    transport_uncertain = np.copy(transport)
    transport_uncertain[~l_uncertain] = 0.0
    
    _, transport_correct_m = get_monthly_sums(time, transport_correct)
    _, transport_incorrect_m = get_monthly_sums(time, transport_incorrect)
    _, transport_uncertain_m = get_monthly_sums(time, transport_uncertain)
    
    str_time = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    
    fig = plt.figure(figsize=(10, 5))
    plt.subplots_adjust(wspace=0.4)
    
    # Tests
    ax1 = plt.subplot(1, 2, 1)
    plot_monthly_histogram(time_m, counts_m,
                           ylabel='Tests (#)',
                           time_is_center=True, color='#25419e',
                           ax=ax1, show=False)
    ax1.set_xticklabels(str_time)
    ylim1 = np.ceil(1.3*max(counts_m))
    ax1.set_ylim([0, ylim1])
    # ax1.set_yticks(np.arange(0, 200, 20))
    ax1.set_xlim([datetime(2017, 1, 1), datetime(2017, 12, 31)])
    add_subtitle(ax1, '(a) Number of tests per month')
    
    ax2 = plt.subplot(1, 2, 2)
    
    center_time = time_m
    time_plus = np.append(time_m, add_month_to_time(time_m[-1], 1))
    width = 0.8*np.array([dt.days for dt in np.diff(time_plus)])
    
    y_scale = 10**4
    
    bottom = np.zeros(len(time_m))
    ax2.bar(time_m, transport_correct_m / y_scale, width, label='DSWT', bottom=bottom, color='#25419e')
    bottom += transport_correct_m / y_scale
    ax2.bar(time_m, transport_incorrect_m / y_scale, width, label='No DSWT', bottom=bottom, color='#900C3F')
    bottom += transport_incorrect_m / y_scale
    ax2.bar(time_m, transport_uncertain_m / y_scale, width, label='Possible', bottom=bottom, color='#929292')
    
    ax2.set_xticks(center_time)
    ax2.set_xticklabels(str_time)
    ax2.set_ylabel('Monthly transport (10$^4$ m$^2$ s$^{-1}$)')
    
    ylim2 = np.ceil(max(transport_correct_m + transport_incorrect_m + transport_uncertain_m) / y_scale / 50) * 50
    ax2.set_ylim([0, ylim2])
    ax2.set_xlim([datetime(2017, 1, 1), datetime(2017, 12, 31)])
    
    ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 0.94))
    
    add_subtitle(ax2, '(b) Monthly effect on transport')
    
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        log.info(f'Saved performance summary to: {output_path}')
        
        data = np.array([transport_correct_m, transport_incorrect_m, transport_uncertain_m]).transpose()
        df_out = pd.DataFrame(data=data, columns=['correct transport (m2/s)', 'incorrect transport (m2/s)', 'uncertain transport (m2/s)'],
                              index=str_time)
        output_csv = f'{os.path.splitext(output_path)[0]}.csv'
        df_out.to_csv(output_csv)
        
    if show == True:
        plt.show()
    else:
        plt.close()

def recheck_differences(input_dir:str, grid_file:str, transects:dict,
                        df:pd.DataFrame):

    manual_dswt = df['manual_dswt'].values
    algorithm_dswt = df['algorithm_dswt'].values

    l_comparison = manual_dswt == algorithm_dswt
    
    df_diff = df.loc[l_comparison == False]
    
    changes = 0
    for filename in np.unique(df_diff['filename'].values):
        input_path = f'{input_dir}{filename}.nc'
        roms_ds = load_roms_data(input_path, grid_file=grid_file)
        roms_times = pd.to_datetime(roms_ds.ocean_time.values)
        
        l_file = df_diff['filename'].values == filename
        df_diff_file = df_diff.loc[l_file]
        
        for i in range(len(df_diff_file)):
            time_str = str(df_diff_file['time'].values[i])
            transect_name = df_diff_file['transect'].values[i]

            time = datetime.strptime(time_str, '%Y%m%d%H%M')
            t = np.where(roms_times == time)[0][0]
            
            eta = transects[transect_name]['eta']
            xi = transects[transect_name]['xi']
            transect_ds = select_roms_transect_from_known_coordinates(roms_ds, eta, xi)
            transects_plot(transect_ds, t)
            plt.show()
            
            manual_input = input('DSWT 0=False, 1=True, 0.5=Possible: ')
            if manual_input != df_diff_file['manual_dswt'].values[i]:
                l_row = np.logical_and(df['filename'] == filename, df['transect'] == transect_name)
                l_col = df.columns == 'manual_dswt'
                df.loc[l_row, l_col] = manual_input
                changes += 1
            
    # write performance comparison to file again if any changes
    if changes > 0:
        df.to_csv(performance_file, index=False)
    else:
        log.info(f'Performance not changed after manual checks of differences')
        
if __name__ == '__main__':
    recheck = True

    year = 2017
    model = 'cwa'
    grid_file = f'{get_dir_from_json("cwa")}grid.nc'
    input_dir = f'{get_dir_from_json("cwa")}{year}/'

    lon_range = [114.0, 116.0]
    lat_range = [-33.0, -31.0]
    transects_file = f'input/transects/{model}_transects.json'
    transects = read_transects_in_lon_lat_range_from_json(transects_file, lon_range, lat_range)

    performance_file = f'performance_tests/output/{model}/{model}_{year}_performance_comparison.csv'
    df = pd.read_csv(performance_file)
    
    if recheck == True:
        recheck_differences(input_dir, grid_file, transects, df)
    
    plot_performance_summary(df, output_path=f'performance_tests/output/{model}/{model}_performance_summary.jpg')
    
    plot_monthly_performance(df, output_path=f'performance_tests/output/{model}/{model}_performance_monthly.jpg')
