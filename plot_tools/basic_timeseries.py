import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.timeseries import add_month_to_time

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
import numpy as np
from datetime import datetime, date, timedelta

converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[date] = converter
munits.registry[datetime] = converter

locator = mdates.AutoDateLocator(minticks=5, maxticks=15)
formatter = mdates.ConciseDateFormatter(locator)

ocean_blue = '#25419e'

def _get_center_label_width_for_bar_plot(time:np.ndarray[datetime], label_format='%b') -> tuple:
    time_plus = np.append(time, add_month_to_time(time[-1], 1))
    center_time = np.array(time+np.diff(time_plus)/2)
    str_time = np.array([t.strftime(label_format) for t in time])
    width = 0.8*np.array([dt.days for dt in np.diff(time_plus)])
    return center_time, str_time, width

def plot_histogram_monthly_dswt_multiple_years(time:np.ndarray[datetime], f_dswt:np.ndarray[float],
                                               ax=None, show=True) -> plt.axes:
    time_years = np.array([t.year for t in time])
    years = np.unique(time_years)
    
    center_time = np.array([])
    width = np.array([])
    for year in years:
        l_year = time_years == year
        center, _, w = _get_center_label_width_for_bar_plot(time[l_year])
        center_time = np.concatenate((center_time, center))
        width = np.concatenate((width, w))
       
    if ax is None:
        ax = plt.axes()
         
    ax.bar(center_time, f_dswt*100, color=ocean_blue, width=width)
    
    ax.set_ylabel('DSWT occurrence (%)')
    ax.set_ylim([0, 100])
    
    ax.set_xticks([datetime(y, 7, 2) for y in years])
    ax.set_xticklabels(years)
    plt.tick_params(axis='x', length=0)
    ax.set_xlim([time[0], time[-1]])
    
    if show == True:
        plt.show()
    else:
        return ax

def plot_histogram_monthly_dswt(time:np.ndarray[datetime], f_dswt:np.ndarray[float],
                                ax=None, show=True) -> plt.axes:
    
    center_time, str_time, width = _get_center_label_width_for_bar_plot(time)
    
    if ax is None:
        ax = plt.axes()
        
    ax.bar(center_time, f_dswt*100, color=ocean_blue, tick_label=str_time, width=width)
    ax.set_ylabel('DSWT occurrence (%)')
    ax.set_ylim([0, 100])
    
    if show is True:
        plt.show()
    else:
        return ax