import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from readers.read_climate_indices import read_mei_data
from readers.surface_fluxes import read_surface_fluxes_from_csvs
from readers.read_meteo_data import read_wind_from_csvs
from tools.timeseries import get_monthly_means, get_monthly_sums, get_yearly_means, get_yearly_sums

import numpy as np
import pandas as pd
from datetime import datetime

def get_domain_str(lon_range:list, lat_range:list) -> str:
    lon_range_str = f'{int(np.floor(lon_range[0]))}-{int(np.ceil(lon_range[1]))}'
    lon_range_unit = 'E' if lon_range[0] > 0 else 'W'
    lat_range_str = f'{int(abs(np.floor(lat_range[0])))}-{int(abs(np.ceil(lat_range[1])))}'
    lat_range_unit = 'S' if lat_range[0] < 0 else 'S'
    domain = f'{lon_range_str}{lon_range_unit}_{lat_range_str}{lat_range_unit}'
    
    return domain

def get_input_paths(input_dir:str, path_str:str, years:list) -> list[str]:
    input_paths = []
    for i in range(len(years)):
        input_paths.append(f'{input_dir}{path_str}_{years[i]}.csv')
        
    return input_paths

def add_total_transport_to_df(df:pd.DataFrame, transects:dict) -> np.ndarray:
    ts = df.loc[df['time']==df['time'][0]]['transect'].values
    widths = np.array([transects[t]['width'] for t in ts])
    widths_all = np.tile(widths, len(pd.unique(df['time'])))    
    df['total_transport'] = df['transport_dswt'].values*widths_all
    return df

def calculate_yearly_transport_per_transect(input_path:str) -> np.ndarray:
    df = pd.read_csv(input_path)
    df_transects = df.set_index(['time', 'transect']).groupby('transect').sum()
    return df_transects.index.values, df_transects['transport_dswt'].values

def calculate_total_daily_transport(input_path:str, transects:dict) -> np.ndarray:
    df = pd.read_csv(input_path)
    df = add_total_transport_to_df(df, transects)

    df_daily_sum = df.set_index(['time', 'transect']).groupby('time').sum()
    daily_transport = df_daily_sum['total_transport'].values
    return daily_transport

def read_dswt_output(input_dir:str, years:list, transects:dict, path_str='dswt') -> tuple:
    # add check to see if number of lines in each file is as expected and raise warning if not
    input_paths = get_input_paths(input_dir, path_str, years)
    time = np.array([])
    f_dswt = np.array([])
    vel_dswt = np.array([])
    transport_dswt = np.array([])
    for input_path in input_paths:
        df = pd.read_csv(input_path, index_col=['time', 'transect'])
        df_daily_mean = df.groupby('time').mean()
        time = np.concatenate((time, np.array([datetime.strptime(t, '%Y-%m-%d') for t in df_daily_mean.index.values])))
        f_dswt = np.concatenate((f_dswt, df_daily_mean['f_dswt'].values))
        vel_dswt = np.concatenate((vel_dswt, df_daily_mean['vel_dswt'].values))
        # calculate total transport (x transect width) separately
        transport_dswt_i = calculate_total_daily_transport(input_path, transects)
        transport_dswt = np.concatenate((transport_dswt, transport_dswt_i))
    
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

def get_monthly_yearly_mei_data(years:list):
    year_range = [years[0], years[-1]]
    time, mei = read_mei_data(year_range=year_range)
    time_y, mei_y = get_yearly_means(time, mei)
    return time, mei, time_y, mei_y

def get_sflux_data(input_dir:str, years:list, path_str='sflux') -> tuple:
    input_paths = get_input_paths(input_dir, path_str, years)
    time, shflux, ssflux = read_surface_fluxes_from_csvs(input_paths)
    
    return time, shflux, ssflux

def get_wind_data(input_dir:str, years:list, path_str='wind') -> tuple:
    input_paths = get_input_paths(input_dir, path_str, years)
    time, u, v, vel, dir = read_wind_from_csvs(input_paths)
    return time, u, v, vel, dir

def get_monthly_atmosphere_data(time:np.ndarray, y1:np.ndarray, y2:np.ndarray) -> tuple:
    time_m, y1_m = get_monthly_means(time, y1)
    _, y2_m = get_monthly_means(time, y2)
    return time_m, y1_m, y2_m

def get_yearly_atmosphere_data(time:np.ndarray, y1:np.ndarray, y2:np.ndarray) -> tuple:
    time_y, y1_y = get_yearly_means(time, y1)
    _, y2_y = get_yearly_means(time, y2)
    return time_y, y1_y, y2_y
