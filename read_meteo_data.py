
from tools.files import get_files_in_dir
from tools.wind import convert_u_v_to_meteo_vel_dir, get_lon_lat_range_indices
from tools.timeseries import get_daily_means
from tools.arrays import get_closest_index
from tools import log
import numpy as np
import xarray as xr
import pandas as pd

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
        subset_ds = subset_ds.sel(lon=slice(i0, i1), lat=slice(j0, j1))
    
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
