import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.files import get_files_in_dir, get_dir_from_json
from tools.wind import convert_u_v_to_meteo_vel_dir, get_lon_lat_range_indices
from tools.timeseries import get_daily_means, get_monthly_means
from tools.arrays import get_closest_index
from tools import log
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime

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

def select_era5_subset_along_coordinates(era5_ds:xr.Dataset,
                                         lon:np.ndarray[float],
                                         lat:np.ndarray[float]):
        
    i = get_closest_index(era5_ds.lon.values[0, :], lon)
    j = get_closest_index(era5_ds.lat.values[:, 0], lat)
    
    i_coords = xr.DataArray(i, dims='distance') # conversion to xr.DataArray needed to select individual points (rather than grid)
    j_coords = xr.DataArray(j, dims='distance') # naming dimension "distance" here allows coordinate values to be linked to it later
    era5_ds_coords = era5_ds.sel(dim1=j_coords, dim2=i_coords)

    return era5_ds_coords

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

class WindTimeseries:
    def __init__(self, time:np.ndarray[datetime],
                 u:np.ndarray[float],
                 v:np.ndarray[float],
                 vel:np.ndarray[float],
                 dir:np.ndarray[float]):
        self.time = time
        self.u = u
        self.v = v
        self.vel = vel
        self.dir = dir
        self.time_m, self.vel_m = get_monthly_means(time, vel)
        _, self.dir_m = get_monthly_means(time, dir)
