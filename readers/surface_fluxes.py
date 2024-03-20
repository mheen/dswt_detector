from readers.read_ocean_data import select_input_files, read_roms_data, select_roms_subset
from tools import log
from tools.files import get_dir_from_json
from datetime import datetime
import numpy as np
import xarray as xr
import pandas as pd
import os

def write_daily_mean_sflux_to_csv(input_dir:str,
                                   file_contains:str,
                                  grid_file:str,
                                  output_path:str,
                                  lon_range:list, lat_range:list):
    
    input_files = select_input_files(input_dir, file_contains=file_contains)
    input_files.sort()
    
    for file in input_files:
        ds = read_roms_data(file, grid_file=grid_file)
        ds_ss = select_roms_subset(ds, None, lon_range, lat_range)
        
        ocean_time0 = pd.to_datetime(ds.ocean_time.values[0])
        time = datetime(ocean_time0.year, ocean_time0.month, ocean_time0.day)
        shflux = np.nanmean(ds_ss.shflux.values)
        ssflux = np.nanmean(ds_ss.ssflux.values)
        data = np.expand_dims(np.array([time, shflux, ssflux]), axis=0)
        df = pd.DataFrame(data, columns=['time', 'shflux', 'ssflux'])
        
        if os.path.exists(output_path):
            df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            df.to_csv(output_path, index=False)

def read_surface_fluxes_from_csvs(input_paths:list[str]) -> tuple[np.ndarray[datetime], np.ndarray[float]]:
    time = pd.to_datetime(np.array([]))
    shflux = np.array([])
    ssflux = np.array([])
    for input_path in input_paths:
        df = pd.read_csv(input_path)
        time_y = pd.to_datetime(df['time'].values)
        shflux_y = df['shflux'].values
        ssflux_y = df['ssflux'].values
        
        time = np.concatenate((time, time_y))
        shflux = np.concatenate((shflux, shflux_y))
        ssflux = np.concatenate((ssflux, ssflux_y))
        
    time = np.array([pd.to_datetime(t) for t in time])
        
    return time, shflux, ssflux
 
if __name__ == '__main__':
    main_input_dir = get_dir_from_json('cwa-roms')
    subdir = 'shflux'
    grid_file = f'{main_input_dir}grid.nc'
    years = np.arange(2000, 2023)
    
    lon_range = [114.0, 116.0]
    lat_range = [-33.0, -31.0]
    
    # output file domain str
    lon_range_str = f'{int(np.floor(lon_range[0]))}-{int(np.ceil(lon_range[1]))}'
    lon_range_unit = 'E' if lon_range[0] > 0 else 'W'
    lat_range_str = f'{int(abs(np.floor(lat_range[0])))}-{int(abs(np.ceil(lat_range[1])))}'
    lat_range_unit = 'S' if lat_range[0] < 0 else 'S'
    domain = f'{lon_range_str}{lon_range_unit}_{lat_range_str}{lat_range_unit}'
    
    for year in years:
        input_dir = f'{main_input_dir}{year}/{subdir}/'
        output_path = f'output/sflux/sflux_cwa_{year}_{domain}.csv'
        write_daily_mean_sflux_to_csv(input_dir, 'cwa_', grid_file, output_path,lon_range, lat_range)
