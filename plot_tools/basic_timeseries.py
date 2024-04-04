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

def _get_center_label_width_for_monthly_bar_plot(time:np.ndarray[datetime], label_format='%b') -> tuple:
    time_plus = np.append(time, add_month_to_time(time[-1], 1))
    center_time = np.array(time+np.diff(time_plus)/2)
    str_time = np.array([t.strftime(label_format) for t in time])
    width = 0.8*np.array([dt.days for dt in np.diff(time_plus)])
    return center_time, str_time, width

def _get_center_label_width_for_multi_monthly_bar_plot(time:np.ndarray[datetime], n_bars:int, label_format='%b') -> tuple:
    time = np.array([datetime(t.year, t.month, 1) for t in time]) # in case time is centered to middle of a month
    time_plus = np.append(time, add_month_to_time(time[-1], 1))
    dt = np.diff(time_plus)
    str_time = np.array(time+dt/2)
    str_labels = np.array([t.strftime(label_format) for t in time])
    center_times = []
    for n in range(n_bars):
        center_times.append(np.array(time+(n+1)/n_bars*dt-1/n_bars*dt/2))
    width = 0.8*np.array([1/n_bars*t.days for t in dt])
    return center_times, width, str_time, str_labels

def _get_center_label_width_for_multi_year_bar_plot(time:np.ndarray[datetime], n_bars:int) -> tuple:
    time = np.array([datetime(t.year, 1, 1) for t in time])
    time_plus = np.append(time, datetime(time[-1].year+1, 1, 1))
    dt = np.diff(time_plus)
    str_time = np.array(time+dt/2)
    str_labels = np.array([t.year for t in time])
    center_times = []
    for n in range(n_bars):
        center_times.append(np.array(time+(n+1)/n_bars*dt-1/n_bars*dt/2))
    width = 0.8*np.array([1/n_bars*t.days for t in dt])
    return center_times, width, str_time, str_labels

def plot_yearly_grid(ax:plt.axes, years:list) -> plt.axes:
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

def plot_histogram_multiple_years(time:np.ndarray[datetime], values:np.ndarray[float],
                                  ylabel='', ylim=None,
                                  color='#25419e', c_change=None,
                                  ax=None, show=True) -> plt.axes:
    time_years = np.array([t.year for t in time])
    years = np.unique(time_years)
    
    center_time = np.array([])
    width = np.array([])
    for year in years:
        l_year = time_years == year
        if np.sum(l_year) == 1:
            center = np.array([datetime(time[l_year][0].year, 7, 2)])
            w = np.array([0.8*365])
        else:
            center, _, w = _get_center_label_width_for_monthly_bar_plot(time[l_year])
        center_time = np.concatenate((center_time, center))
        width = np.concatenate((width, w))
       
    if ax is None:
        ax = plt.axes()

    if type(color) == str:
        ax.bar(center_time, values, color=color, width=width)
    elif type(color) == list and c_change is not None:
        l0 = values <= c_change
        l1 = values > c_change
        ax.bar(center_time[l0], values[l0], color=color[0], width=width[l0])
        ax.bar(center_time[l1], values[l1], color=color[1], width=width[l1])
        ax.plot([center_time[0], center_time[-1]], [c_change, c_change], '-k')
    
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ylim = ax.get_ylim()
    
    ax.set_xticks([datetime(y, 7, 2) for y in years]) # ticks in the middle of the year
    ax.set_xticklabels(years, rotation='vertical')
    plt.tick_params(axis='x', length=0)
    ax.set_xlim([time[0], time[-1]])
    
    for y in years: # plot grid to show years
        ax.plot([datetime(y, 1, 1), datetime(y, 1, 1)], ylim, '-', color='#808080', alpha=0.2)
    
    if show == True:
        plt.show()
    else:
        return ax

def plot_monthly_histogram(time:np.ndarray[datetime], values:np.ndarray[float],
                           ylabel='', ylim=None, time_is_center=False,
                           color='#25419e', c_change=None,
                           ax=None, show=True) -> plt.axes:
    
    if time_is_center is False:
        center_time, str_time, width = _get_center_label_width_for_monthly_bar_plot(time)
    else:
        center_time = time
        time_plus = np.append(time, add_month_to_time(time[-1], 1))
        str_time = np.array([t.strftime('%b') for t in time])
        width = 0.8*np.array([dt.days for dt in np.diff(time_plus)])
    
    if ax is None:
        ax = plt.axes()
        
    if type(color) == str:
        ax.bar(center_time, values, color=color, width=width)
    elif type(color) == list and c_change is not None:
        l0 = values <= c_change
        l1 = values > c_change
        ax.bar(center_time[l0], values[l0], color=color[0], width=width[l0])
        ax.bar(center_time[l1], values[l1], color=color[1], width=width[l1])
        ax.plot([center_time[0], center_time[-1]], [c_change, c_change], '-k')
    
    ax.set_xticks(center_time)
    ax.set_xticklabels(str_time)
    
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if show is True:
        plt.show()
    else:
        return ax, center_time, str_time
    
def plot_multi_bar_monthly_histogram(time:np.ndarray[datetime], values:list[np.ndarray[float]], colors:list,
                                     labels:list, ylabel='', ylim=None, legend_loc='upper right',
                                     ax=None, show=True) -> plt.axes:
    center_times, width, str_time, str_labels = _get_center_label_width_for_multi_monthly_bar_plot(time, len(values))
    
    if ax is None:
        ax = plt.axes()
        
    for i in range(len(values)):
        ax.bar(center_times[i], values[i], color=colors[i], width=width, label=labels[i])
    
    ax.set_xticks(str_time)
    ax.set_xticklabels(str_labels)
    
    l = ax.legend(loc=legend_loc)
    
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if show is True:
        plt.show()
    else:
        return ax, l

def plot_multi_bar_yearly_histogram(time:np.ndarray[datetime], values:list[np.ndarray[float]], colors:list,
                                     labels:list, ylabel='', ylim=None, legend_loc='upper right',
                                     ax=None, show=True) -> plt.axes:
    center_times, width, str_time, str_labels = _get_center_label_width_for_multi_year_bar_plot(time, len(values))
    
    if ax is None:
        ax = plt.axes()
        
    for i in range(len(values)):
        ax.bar(center_times[i], values[i], color=colors[i], width=width, label=labels[i])
    
    ax.set_xticks(str_time)
    ax.set_xticklabels(str_labels)
    
    l = ax.legend(loc=legend_loc)
    
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if show is True:
        plt.show()
    else:
        return ax, l