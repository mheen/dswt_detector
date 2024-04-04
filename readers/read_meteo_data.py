import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.files import get_files_in_dir, get_dir_from_json
from tools.wind import convert_u_v_to_meteo_vel_dir, get_lon_lat_range_indices
from tools.timeseries import get_daily_means
from tools.arrays import get_closest_index
from tools import log
import numpy as np
import xarray as xr
import pandas as pd
import dask
from dask.distributed import Client
from datetime import datetime

# c = Client(n_workers=8, threads_per_worker=1, memory_limit='2GB')
# c.cluster

def select_input_paths(input_dir:str, file_contains:str, filetype='nc') -> list:
    all_files = get_files_in_dir(input_dir, filetype)
    if file_contains is not None:
        files = [f for f in all_files if file_contains in f]
    else:
        files = all_files
        
    return files

def load_era5_data(input_dir:list, file_contains:str) -> xr.DataArray:
    input_paths = select_input_paths(input_dir, file_contains)
    era5_ds = xr.open_mfdataset(input_paths, data_vars='minimal')
    
    vel, dir = convert_u_v_to_meteo_vel_dir(era5_ds.Uwind.values, era5_ds.Vwind.values)
    
    era5_ds['wind_vel'] = (['time', 'dim1', 'dim2'], vel)
    era5_ds['wind_dir'] = (['time', 'dim1', 'dim2'], dir)
    
    return era5_ds

def select_era5_subset(era5_ds:xr.Dataset,
                       time_range:list,
                       lon_range:list,
                       lat_range:list) -> xr.Dataset:
    
    if isinstance(time_range, list):
        subset_ds = era5_ds.sel(time=slice(time_range[0], time_range[1]))
    elif isinstance(time_range, str):
        subset_ds = era5_ds.sel(time=time_range)
    else:
        subset_ds = era5_ds
    
    if lon_range is not None and lat_range is not None:
        i0, i1, j0, j1 = get_lon_lat_range_indices(era5_ds.lon.values, era5_ds.lat.values, lon_range, lat_range)
        subset_ds = subset_ds.isel(dim2=slice(i0, i1), dim1=slice(j0, j1))
    
    return subset_ds

def select_era5_in_closest_point(era5_ds:xr.Dataset,
                                 lon_p:float,
                                 lat_p:float):
    i = get_closest_index(era5_ds.lon.values[0, :], lon_p)
    j = get_closest_index(era5_ds.lat.values[:, 0], lat_p)
    point_ds = era5_ds.isel(dim1=j, dim2=i)
    
    return point_ds

def get_daily_mean_wind_data(era5_ds:xr.Dataset) -> tuple:
    time = pd.to_datetime(era5_ds.time.values)
    dm_time, dm_u = get_daily_means(time, era5_ds.Uwind.values)
    _, dm_v = get_daily_means(time, era5_ds.Vwind)

    dm_vel, dm_dir = convert_u_v_to_meteo_vel_dir(dm_u, dm_v)

    return (dm_time, dm_u, dm_v, dm_vel, dm_dir)

def get_daily_mean_wind_data_over_area(era5_ds:xr.Dataset) -> tuple:
    time = pd.to_datetime(era5_ds.time.values)
    mean_u = np.nanmean(np.nanmean(era5_ds.Uwind.values, axis=1), axis=1)
    mean_v = np.nanmean(np.nanmean(era5_ds.Vwind.values, axis=1), axis=1)
    dm_time, dm_u = get_daily_means(time, mean_u)
    _, dm_v = get_daily_means(time, mean_v)

    dm_vel, dm_dir = convert_u_v_to_meteo_vel_dir(dm_u, dm_v)

    return (dm_time, dm_u, dm_v, dm_vel, dm_dir)

def write_daily_means_to_csv(input_dir:str, year:float,
                             lon_range:list, lat_range:list,
                             output_path:str):
    ds = load_era5_data(input_dir, datetime(year, 1, 1).strftime("%Y%m%d"))
    ds_ss = select_era5_subset(ds, None, lon_range, lat_range)
    
    time, u, v, vel, dir = get_daily_mean_wind_data_over_area(ds_ss)
    data = np.array([time, u, v, vel, dir]).transpose()
    df = pd.DataFrame(data, columns=['time', 'u', 'v', 'vel', 'dir'])
    df.to_csv(output_path, index=False)

def read_wind_from_csvs(input_paths:list[str]) -> tuple[np.ndarray[datetime], np.ndarray[float]]:
    time = pd.to_datetime(np.array([]))
    u = np.array([])
    v = np.array([])
    vel = np.array([])
    dir = np.array([])
    for input_path in input_paths:
        df = pd.read_csv(input_path)
        time_y = pd.to_datetime(df['time'].values)
        u_y = df['u'].values
        v_y = df['v'].values
        vel_y = df['vel'].values
        dir_y = df['dir'].values
        
        time = np.concatenate((time, time_y))
        u = np.concatenate((u, u_y))
        v = np.concatenate((v, v_y))
        vel = np.concatenate((vel, vel_y))
        dir = np.concatenate((dir, dir_y))
        
    time = np.array([pd.to_datetime(t) for t in time])
        
    return time, u, v, vel, dir
    
if __name__ == '__main__':
    input_dir = get_dir_from_json('era5')
    years = np.arange(2001, 2023)
    
    lon_range = [114.0, 116.0]
    lat_range = [-33.0, -31.0]
    
    # output file domain str
    lon_range_str = f'{int(np.floor(lon_range[0]))}-{int(np.ceil(lon_range[1]))}'
    lon_range_unit = 'E' if lon_range[0] > 0 else 'W'
    lat_range_str = f'{int(abs(np.floor(lat_range[0])))}-{int(abs(np.ceil(lat_range[1])))}'
    lat_range_unit = 'S' if lat_range[0] < 0 else 'S'
    domain = f'{lon_range_str}{lon_range_unit}_{lat_range_str}{lat_range_unit}'
    
    for year in years:
        output_path = f'output/cwa_{domain}/wind/wind_{year}.csv'
        write_daily_means_to_csv(input_dir, year, lon_range, lat_range, output_path)