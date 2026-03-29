import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from readers.read_ocean_data import get_roms_contour_coordinates, get_roms_ds_along_contour

from tools.roms import get_eta_xi_of_lon_lat_point
from tools.coordinates import get_distance_between_points
from tools.files import get_dir_from_json
from tools.timeseries import get_time_indices
from tools import log

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import os

def read_df_from_multiple_csvs(input_dir:str, years:list, file_preface:str) -> pd.DataFrame:
    df = None
    
    for year in years:
        input_path = f'{input_dir}{file_preface}{year}.csv'
        if not os.path.exists(input_path):
            continue
        
        df_y = pd.read_csv(input_path)
        
        if df is None:
            df = df_y
            continue
        
        df = pd.concat([df, df_y], ignore_index=True)
        
    return df

def get_domain_str(lon_range:list, lat_range:list) -> str:
    lon_range_str = f'{int(np.floor(lon_range[0]))}-{int(np.ceil(lon_range[1]))}'
    lon_range_unit = 'E' if lon_range[0] > 0 else 'W'
    lat_range_str = f'{int(abs(np.floor(lat_range[0])))}-{int(abs(np.ceil(lat_range[1])))}'
    lat_range_unit = 'S' if lat_range[0] < 0 else 'S'
    domain = f'{lon_range_str}{lon_range_unit}_{lat_range_str}{lat_range_unit}'
    
    return domain

def _create_daily_time_array(year_start:int, year_end:int) -> np.ndarray[datetime]:
    time = []
    start_date = datetime(year_start, 1, 1)
    end_date = datetime(year_end, 12, 31)
    n_days = (end_date-start_date).days + 1
    for n in range(n_days):
        time.append(start_date + timedelta(days=n))
    time = np.array(time)
    return time

def get_dswt_daily_timeseries_df(input_dir:str, years:list) -> tuple[np.ndarray[datetime], np.ndarray[float]]:
    '''Reads DSWT detection output from multiple files for multiple years (if needed) and determines daily
    DSWT measures over all transects.
    
    Input:
    - input_dir [string]: path to directory that contains output csv files from the DSWT detection
    - years [list]: list of years for which to read DSWT occurrence
      note: if you only want to read a single year, put that year into a list
    
    Output:
    - df: DataFrame with daily measures'''
    
    time = np.array([])
    f_dswt = np.array([])
    transport = np.array([])
    thickness = np.array([])
    vel = np.array([])
    drhodx = np.array([])
    min_distance = np.array([])
    max_distance = np.array([])
    min_h = np.array([])
    mean_h = np.array([])
    max_h = np.array([])
    drho_z = np.array([])
    min_drho_z = np.array([])
    
    for year in years:
        input_path = f'{input_dir}dswt_{year}.csv'
        time_year = _create_daily_time_array(year, year)

        if os.path.exists(input_path) == False:
            log.info(f'Input file does not exist, adding empty array instead: {input_path}')
            time = np.concatenate((time, time_year))
            f_dswt = np.concatenate((f_dswt, np.empty(len(time_year))*np.nan))
            transport = np.concatenate((transport, np.empty(len(time_year))*np.nan))
            thickness = np.concatenate((thickness, np.empty(len(time_year))*np.nan))
            vel = np.concatenate((vel, np.empty(len(time_year))*np.nan))
            drhodx = np.concatenate((drhodx, np.empty(len(time_year))*np.nan))
            min_distance = np.concatenate((min_distance, np.empty(len(time_year))*np.nan))
            max_distance = np.concatenate((max_distance, np.empty(len(time_year))*np.nan))
            min_h = np.concatenate((min_h, np.empty(len(time_year))*np.nan))
            mean_h = np.concatenate((mean_h, np.empty(len(time_year))*np.nan))
            max_h = np.concatenate((max_h, np.empty(len(time_year))*np.nan))
            drho_z = np.concatenate((drho_z, np.empty(len(time_year))*np.nan))
            min_drho_z = np.concatenate((min_drho_z, np.empty(len(time_year))*np.nan))
            continue
        
        df = pd.read_csv(input_path)
        df_daily_mean = (df.groupby(['time', 'transect']).mean()).groupby('time').mean() # group by both time and transect because there are multiple values for transect
        df_daily_min = (df.groupby(['time', 'transect']).min()).groupby('time').min()
        df_daily_max = (df.groupby(['time', 'transect']).max()).groupby('time').max()
        
        # f_dswt with zero values included
        time_org = np.array([datetime.strptime(t, '%Y-%m-%d') for t in df_daily_mean.index.values])
        f_dswt_year = np.zeros(len(time_year))
        transport_year = np.empty(len(time_year)) * np.nan
        thickness_year = np.empty(len(time_year)) * np.nan
        vel_year = np.empty(len(time_year)) * np.nan
        drhodx_year = np.empty(len(time_year)) * np.nan
        min_distance_year = np.empty(len(time_year)) * np.nan
        max_distance_year = np.empty(len(time_year)) * np.nan
        min_h_year = np.empty(len(time_year)) * np.nan
        mean_h_year = np.empty(len(time_year)) * np.nan
        max_h_year = np.empty(len(time_year)) * np.nan
        drho_z_year = np.empty(len(time_year)) * np.nan
        min_drho_z_year = np.empty(len(time_year)) * np.nan
        for t in range(len(time_org)):
            i_time = np.where(time_org[t] == time_year)[0][0]
            f_dswt_year[i_time] = df_daily_mean['f_dswt'].values[t]
            transport_year[i_time] = df_daily_mean['transport'].values[t]
            thickness_year[i_time] = df_daily_mean['thickness'].values[t]
            vel_year[i_time] = df_daily_mean['vel'].values[t]
            drhodx_year[i_time] = df_daily_mean['drhodx_mean'].values[t]
            min_distance_year[i_time] = df_daily_min['distance'].values[t]
            max_distance_year[i_time] = df_daily_max['distance'].values[t]
            min_h_year[i_time] = df_daily_min['h'].values[t]
            mean_h_year[i_time] = df_daily_mean['h'].values[t]
            max_h_year[i_time] = df_daily_max['h'].values[t]
            drho_z_year[i_time] = df_daily_mean['drho_z_mean'].values[t]
            min_drho_z_year[i_time] = df_daily_min['drho_z_min'].values[t]
        
        time = np.concatenate((time, time_year))
        f_dswt = np.concatenate((f_dswt, f_dswt_year))
        transport = np.concatenate((transport, transport_year))
        thickness = np.concatenate((thickness, thickness_year))
        vel = np.concatenate((vel, vel_year))
        drhodx = np.concatenate((drhodx, drhodx_year))
        min_distance = np.concatenate((min_distance, min_distance_year))
        max_distance = np.concatenate((max_distance, max_distance_year))
        min_h = np.concatenate((min_h, min_h_year))
        mean_h = np.concatenate((mean_h, mean_h_year))
        max_h = np.concatenate((max_h, max_h_year))
        drho_z = np.concatenate((drho_z, drho_z_year))
        min_drho_z = np.concatenate((min_drho_z, min_drho_z_year))
    
    data = np.array([time, f_dswt, vel, thickness, transport, drhodx, min_distance, max_distance, min_h, mean_h, max_h, drho_z, min_drho_z])
    columns = ['time', 'f_dswt', 'vel', 'thickness', 'transport', 'drhodx', 'min_distance', 'max_distance', 'min_h', 'mean_h', 'max_h', 'drho_z_mean', 'drho_z_min']
    
    df_timeseries = pd.DataFrame(data=data.transpose(), columns=columns)
    
    return df_timeseries

def read_dswt_transport(input_dir:str, years:list, grid_ds:xr.Dataset) -> xr.Dataset:
    '''Reads DSWT detection output from multiple files for multiple years (if needed) and determines the daily
    cross-shelf transport associated with DSWT occurrence. Transport values are returned as both
    a function of time and location in an xarray dataset.
    
    Input:
    - input_dir [string]: path to directory that contains output csv files from the DSWT detection
    - years [list]: list of years for which to read DSWT occurrence
      note: if you only want to read a single year, put that year into a list
    - grid_file [string]: path to ocean model grid file
    
    Output:
    - df_transport [pd.DataFrame]: daily transport as function of location'''
    
    lon = grid_ds.lon_rho.values
    lat = grid_ds.lat_rho.values
    
    n_days = (datetime(years[-1], 12, 31) - datetime(years[0], 1, 1)).days + 1
    overall_time = []
    for n in range(n_days):
        overall_time.append(datetime(years[0], 1, 1) + timedelta(days=n))
    overall_time = np.array(overall_time)
    
    df_total = pd.DataFrame(columns=['time', 'eta', 'xi', 'transport', 'size', 'mean_thickness', 'mean_vel'])
    
    for i in range(len(years)):
        input_path = f'{input_dir}dswt_{years[i]}.csv'
        if not os.path.exists(input_path):
            continue
        
        df = pd.read_csv(input_path)
        df = df.drop(df[np.isnan(df['transport'])].index)
        df['time'] = pd.to_datetime(df['time'])
        eta, xi = get_eta_xi_of_lon_lat_point(lon, lat, df['lon'].values, df['lat'].values)
        df['eta'] = eta
        df['xi'] = xi
        df_daily_transport_per_loc = df.groupby(['time', 'eta', 'xi']).agg(
            transport=('transport', 'mean'),
            mean_thickness=('thickness', 'mean'),
            mean_vel=('vel', 'mean')).reset_index()

        df_total = pd.concat([df_total, df_daily_transport_per_loc])
            
    return df_total

def get_transport_map(df_transport:pd.DataFrame,
                      l_time:np.ndarray[bool],
                      map_shape:list[int],
                      variable='transport'):

    df_sum = df_transport[l_time].groupby(['eta', 'xi']).agg(values=(variable, 'sum'))
    n_time = len(pd.unique(df_transport[l_time]['time'].values))
    df_map = df_sum / n_time
    
    transport_map = np.empty(map_shape)*np.nan
    for i in range(len(df_map)):
        transport_map[df_map.index.values[i]] = df_map['values'].values[i]
    
    return transport_map

def get_daily_transport_across_contour_df(df_transport:pd.DataFrame,
                                       grid_ds:xr.Dataset,
                                       lon_range:list,
                                       lat_range:list,
                                       depth_contour:float,
                                       dx_method='roms'):
    
    dx = np.sqrt(1/grid_ds.pm.values*1/grid_ds.pn.values)
    
    # get contour coordinates
    l_lon = np.logical_and(grid_ds.lon_rho.values >= lon_range[0], grid_ds.lon_rho.values <= lon_range[1])
    l_lat = np.logical_and(grid_ds.lat_rho.values >= lat_range[0], grid_ds.lat_rho.values <= lat_range[1])
    l_range = np.logical_and(l_lon, l_lat)
    
    h = np.copy(grid_ds.h.values)
    h[~l_range] = np.nan
    
    ax = plt.axes()
    cs = ax.contour(grid_ds.lon_rho.values, grid_ds.lat_rho.values, h, levels=[depth_contour])
    vertices = cs.get_paths()[0].vertices
    lon = np.array([coords[0] for coords in vertices])
    lat = np.array([coords[1] for coords in vertices])
    plt.close()
    
    # contour length
    contour_length = 0
    for i in range(len(lon)-1):
        contour_length += get_distance_between_points(lon[i], lat[i], lon[i+1], lat[i+1])
    
    eta, xi = get_eta_xi_of_lon_lat_point(grid_ds.lon_rho.values, grid_ds.lat_rho.values, lon, lat)
    contour_coords = list(zip(eta, xi))
    contour_coords, i_unique = np.unique(contour_coords, axis=0, return_index=True) # remove double coordinates
    i_sort = np.argsort(i_unique)
    contour_coords = contour_coords[i_sort]
    i_unique = i_unique[i_sort]

    # grid cell sizes
    if dx_method == 'roms':
        dx = dx[eta[i_unique], xi[i_unique]]
    
    elif dx_method == 'coords':
        ds = []
        lon_roms = lon[eta[i_unique], xi[i_unique]]
        lat_roms = lat[eta[i_unique], xi[i_unique]]
        for i in range(len(i_unique)):
            if i == 0:
                ds.append(get_distance_between_points(lon_roms[i], lat_roms[i], lon_roms[i+1], lat_roms[i+1])/2)
            elif i == len(i_unique) - 1:
                ds.append(get_distance_between_points(lon_roms[i-1], lat_roms[i-1], lon_roms[i], lat_roms[i])/2)
            else:
                si = get_distance_between_points(lon_roms[i-1], lat_roms[i-1], lon_roms[i], lat_roms[i])
                si1 = get_distance_between_points(lon_roms[i], lat_roms[i], lon_roms[i+1], lat_roms[i+1])
                ds.append(si/2 + si1/2)
        dx = np.array(ds)
    else:
        raise ValueError(f'Unknown dx method: {dx_method}. Valid options are "roms" or "coords".')
    
    # df along contour
    l_contour = []
    dx_for_in_df = []
    for i in range(len(df_transport)):
        l_eta = np.isin(contour_coords[:, 0], df_transport['eta'].values[i])
        l_xi = np.isin(contour_coords[:, 1], df_transport['xi'].values[i])
        l_eta_xi = np.logical_and(l_eta, l_xi)
        l_on_contour = np.any(l_eta_xi)
        l_contour.append(l_on_contour)
        if l_on_contour == True:
            i_coord = np.where(l_eta_xi)[0][0]
            dx_for_in_df.append(dx[i_coord])

    l_contour = np.array(l_contour).astype(bool)
    df_contour = df_transport[l_contour]
    df_contour['time'] = pd.to_datetime(df_contour['time'].values)
    
    # transport array across contour (including zero values)
    # create time array:
    time = _create_daily_time_array(min(df_contour['time']).year, max(df_contour['time']).year)
    
    # daily transport across contour
    df_contour['dx'] = dx_for_in_df
    df_contour['transport_m3'] = df_contour['transport'].values * df_contour['dx'].values
    df_contour_daily = df_contour.groupby(['time']).agg(total_transport=('transport_m3', 'sum'),
                                                        vel_contour=('mean_vel', 'mean'),
                                                        thickness_contour=('mean_thickness', 'mean'))
    df_contour_daily[f'transport_{str(int(depth_contour))}m'] = df_contour_daily['total_transport'].values / contour_length
    
    i_times = get_time_indices(time, df_contour_daily.index)
    daily_transport_all_days = np.zeros(len(time))
    daily_transport_all_days[i_times] = df_contour_daily[f'transport_{str(int(depth_contour))}m'].values
    daily_vel_all_days = np.zeros(len(time))
    daily_vel_all_days[i_times] = df_contour_daily['vel_contour'].values
    daily_thickness_all_days = np.zeros(len(time))
    daily_thickness_all_days[i_times] = df_contour_daily['thickness_contour'].values

    return time, daily_transport_all_days, daily_vel_all_days, daily_thickness_all_days
