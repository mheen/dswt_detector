import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.timeseries import get_monthly_means, get_monthly_sums, get_yearly_means, get_yearly_sums
from tools.roms import get_eta_xi_of_lon_lat_point
from tools.files import get_dir_from_json

import numpy as np
import pandas as pd
from datetime import datetime
import xarray as xr

def get_domain_str(lon_range:list, lat_range:list) -> str:
    lon_range_str = f'{int(np.floor(lon_range[0]))}-{int(np.ceil(lon_range[1]))}'
    lon_range_unit = 'E' if lon_range[0] > 0 else 'W'
    lat_range_str = f'{int(abs(np.floor(lat_range[0])))}-{int(abs(np.ceil(lat_range[1])))}'
    lat_range_unit = 'S' if lat_range[0] < 0 else 'S'
    domain = f'{lon_range_str}{lon_range_unit}_{lat_range_str}{lat_range_unit}'
    
    return domain

def get_input_paths(input_dir:str, years:list) -> list[str]:
    input_paths = []
    for i in range(len(years)):
        input_paths.append(f'{input_dir}dswt_{years[i]}.csv')
        
    return input_paths

def get_daily_velocity_and_transport_maps(input_path:str,
                                          lon:np.ndarray[float],
                                          lat:np.ndarray[float])-> tuple[np.ndarray[datetime],
                                                                 np.ndarray[float], np.ndarray[float],
                                                                 np.ndarray[float]]:
    df = pd.read_csv(input_path)
    time = pd.unique(df['time']) # unique days

    velocity_dswt = np.zeros((len(time), lon.shape[0], lon.shape[1]))
    transport_dswt = np.zeros((len(time), lon.shape[0], lon.shape[1]))
    count = np.zeros(lon.shape)
    
    for i, t in enumerate(time):
        df_day = df[df['time'] == t]
        lon_t = df_day['lon_transport'].values
        lat_t = df_day['lat_transport'].values
        v_dswt = df_day['vel_dswt']
        t_dswt = df_day['transport_dswt']
        
        l_nonan = np.logical_and(~np.isnan(lon_t), ~np.isnan(lat_t))
        if sum(l_nonan) == 0:
            continue
        
        eta, xi = get_eta_xi_of_lon_lat_point(lon, lat, lon_t[l_nonan], lat_t[l_nonan])
        velocity_dswt[i, eta, xi] += v_dswt[l_nonan]
        transport_dswt[i, eta, xi] += t_dswt[l_nonan]
        count[eta, xi] += 1
        
        velocity_dswt[i, :, :] = velocity_dswt[i, :, :]/count # CHECK: does this go right? 2/0 = inf, 0/0 = nan
        transport_dswt[i, :, :] = transport_dswt[i, :, :]/count
        
    velocity_dswt[velocity_dswt == 0] = np.nan
    transport_dswt[transport_dswt == 0] = np.nan
    
    time = np.array([datetime.strptime(t, '%Y-%m-%d') for t in time])
        
    return time, lon, lat, velocity_dswt, transport_dswt

def read_multifile_timeseries(input_dir:str, years:list) -> tuple[np.ndarray[datetime], np.ndarray[float],
                                                                  np.ndarray[float], np.ndarray[float]]:
    input_paths = get_input_paths(input_dir, years)
    time = np.array([])
    f_dswt = np.array([])
    vel_dswt = np.array([])
    transport_dswt = np.array([])
    
    for input_path in input_paths:
        df = pd.read_csv(input_path)
        df_daily_mean = (df.groupby(['time', 'transect']).mean()).groupby('time').mean() # both time and transect because there are multiple values for transect
        time = np.concatenate((time, np.array([datetime.strptime(t, '%Y-%m-%d') for t in df_daily_mean.index.values])))
        f_dswt = np.concatenate((f_dswt, df_daily_mean['f_dswt'].values))
        vel_dswt = np.concatenate((vel_dswt, df_daily_mean['vel_dswt'].values))
        transport_dswt = np.concatenate((transport_dswt, df_daily_mean.groupby('time').sum()['transport_dswt'].values)) # m2
        # note: transport here is the mean along each transect and then the daily sum over all transects
        
    return time, f_dswt, vel_dswt, transport_dswt

def get_monthly_dswt_values(time:np.ndarray,
                            f_dswt:np.ndarray,
                            vel_dswt:np.ndarray,
                            transport_dswt:np.ndarray) -> tuple:
    time_m, f_dswt_m = get_monthly_means(time, f_dswt)
    _, vel_dswt_m = get_monthly_means(time, vel_dswt)
    _, transport_dswt_m = get_monthly_sums(time, transport_dswt)
    
    return time_m, f_dswt_m, vel_dswt_m, transport_dswt_m

def get_yearly_dswt_values(time:np.ndarray,
                           f_dswt:np.ndarray,
                           vel_dswt:np.ndarray,
                           transport_dswt:np.ndarray) -> tuple:
    time_y, f_dswt_y = get_yearly_means(time, f_dswt)
    _, vel_dswt_y = get_yearly_means(time, vel_dswt)
    _, transport_dswt_y = get_yearly_sums(time, transport_dswt)
    
    return time_y, f_dswt_y, vel_dswt_y, transport_dswt_y
