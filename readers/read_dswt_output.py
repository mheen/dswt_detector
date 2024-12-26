import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.roms import get_eta_xi_of_lon_lat_point
from tools.coordinates import get_distance_between_points
from tools import log

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import os

def _create_daily_time_array(year_start:int, year_end:int) -> np.ndarray[datetime]:
    time = []
    start_date = datetime(year_start, 1, 1)
    end_date = datetime(year_end, 12, 31)
    n_days = (end_date-start_date).days + 1
    for n in range(n_days):
        time.append(start_date + timedelta(days=n))
    time = np.array(time)
    return time

def read_dswt_occurrence_timeseries(input_dir:str, years:list) -> tuple[np.ndarray[datetime], np.ndarray[float]]:
    '''Reads DSWT detection output from multiple files for multiple years (if needed) and determines the daily
    fraction of DSWT occurrence over the fraction of transects.
    
    Input:
    - input_dir [string]: path to directory that contains output csv files from the DSWT detection
    - years [list]: list of years for which to read DSWT occurrence
      note: if you only want to read a single year, put that year into a list
    
    Output:
    - time: numpy datetime array
    - f_dswt: fraction of DSWT occurrence'''
    
    time = np.array([])
    f_dswt = np.array([])
    
    for year in years:
        input_path = f'{input_dir}dswt_{year}.csv'
        time_year = _create_daily_time_array(year, year)

        if os.path.exists(input_path) == False:
            log.info(f'Input file does not exist, adding empty array instead: {input_path}')
            time = np.concatenate((time, time_year))
            f_dswt = np.concatenate((f_dswt, np.empty(len(time_year))*np.nan))
            continue
        
        df = pd.read_csv(input_path)
        df_daily_mean = (df.groupby(['time', 'transect']).mean()).groupby('time').mean() # group by both time and transect because there are multiple values for transect
        
        # f_dswt with zero values included
        time_org = np.array([datetime.strptime(t, '%Y-%m-%d') for t in df_daily_mean.index.values])
        f_dswt_year = np.zeros(len(time_year))
        for t in range(len(time_org)):
            i_time = np.where(time_org[t] == time_year)[0][0]
            f_dswt_year[i_time] = df_daily_mean['f_dswt'].values[t]
        
        time = np.concatenate((time, time_year))
        f_dswt = np.concatenate((f_dswt, f_dswt_year))
        
    return time, f_dswt

def read_dswt_transport(input_dir:str, years:list, grid_file:str) -> tuple:
    '''Reads DSWT detection output from multiple files for multiple years (if needed) and determines the daily
    cross-shelf transport associated with DSWT occurrence. Transport values are returned as both
    a function of time and location.
    
    Input:
    - input_dir [string]: path to directory that contains output csv files from the DSWT detection
    - years [list]: list of years for which to read DSWT occurrence
      note: if you only want to read a single year, put that year into a list
    - grid_file [string]: path to ocean model grid file
    
    Output:
    - time: numpy datetime array
    - f_dswt: fraction of DSWT occurrence'''
    
    grid_ds = xr.load_dataset(grid_file)
    dx = np.sqrt(1/grid_ds.pm.values*1/grid_ds.pn.values)
    
    lon = grid_ds.lon_rho.values
    lat = grid_ds.lat_rho.values
    h = grid_ds.h.values
    
    df_total = pd.DataFrame(columns=['time', 'eta', 'xi', 'transport_dswt', 'size', 'dz_mean', 'ds_max'])
    
    for i in range(len(years)):
        input_path = f'{input_dir}dswt_{years[i]}.csv'
        if not os.path.exists(input_path):
            continue
        
        df = pd.read_csv(input_path)
        df = df.drop(df[np.isnan(df['transport_dswt'])].index)
        df['time'] = pd.to_datetime(df['time'])
        eta, xi = get_eta_xi_of_lon_lat_point(lon, lat, df['lon_transport'].values, df['lat_transport'].values)
        df['eta'] = eta
        df['xi'] = xi
        df_daily_transport_per_loc = df.groupby(['time', 'eta', 'xi']).agg(
            transport_dswt=('transport_dswt', 'mean'),
            size=('transport_dswt', 'size'),
            dz_mean=('dz_dswt', 'mean'),
            ds_max=('ds', 'max')).reset_index()

        df_total = pd.concat([df_total, df_daily_transport_per_loc])
        
    df_total['coords'] = list(zip(df_total['eta'], df_total['xi']))
            
    return lon, lat, h, df_total, dx

def get_transport_map(df_transport:pd.DataFrame,
                      l_time:np.ndarray[bool],
                      map_shape:list[int]):
    df_map = df_transport[l_time].groupby(['coords']).agg(transport_dswt=('transport_dswt', 'mean'))
    
    transport_map = np.empty(map_shape)*np.nan
    for i in range(len(df_map)):
        transport_map[df_map.index.values[i]] = df_map['transport_dswt'].values[i]
        
    return transport_map

def calculate_transport_across_contour(df_transport:pd.DataFrame,
                                       lon_grid:np.ndarray,
                                       lat_grid:np.ndarray,
                                       h:np.ndarray,
                                       dx:np.ndarray,
                                       lon_range:list,
                                       lat_range:list,
                                       depth_contour:float,
                                       dx_method='roms'):
    # get contour coordinates
    l_lon = np.logical_and(lon_grid >= lon_range[0], lon_grid <= lon_range[1])
    l_lat = np.logical_and(lat_grid >= lat_range[0], lat_grid <= lat_range[1])
    l_range = np.logical_and(l_lon, l_lat)
    
    h_wcs = np.copy(h)
    h_wcs[~l_range] = np.nan
    
    ax = plt.axes()
    cs = ax.contour(lon_grid, lat_grid, h_wcs, levels=[depth_contour])
    vertices = cs.get_paths()[0].vertices
    lon = np.array([coords[0] for coords in vertices])
    lat = np.array([coords[1] for coords in vertices])
    plt.close()
    
    # contour length
    contour_length = 0
    for i in range(len(lon)-1):
        contour_length += get_distance_between_points(lon[i], lat[i], lon[i+1], lat[i+1])
    
    eta, xi = get_eta_xi_of_lon_lat_point(lon_grid, lat_grid, lon, lat)
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
    coords = df_transport['coords'].values
    l_contour = []
    dx_for_in_df = []
    for i in range(len(coords)):
        l_eta = np.isin(contour_coords[:, 0], coords[i][0])
        l_xi = np.isin(contour_coords[:, 1], coords[i][1])
        l_eta_xi = np.logical_and(l_eta, l_xi)
        l_on_contour = np.any(l_eta_xi)
        l_contour.append(l_on_contour)
        if l_on_contour == True:
            i_coord = np.where(l_eta_xi)[0][0]
            dx_for_in_df.append(dx[i_coord])

    df_contour = df_transport[l_contour]
    df_contour['time'] = pd.to_datetime(df_contour['time'].values)
    
    # transport array across contour (including zero values)
    # create time array:
    time = _create_daily_time_array(min(df_contour['time']).year, max(df_contour['time']).year)
    
    # daily transport across contour
    df_contour['dx'] = dx_for_in_df
    df_contour['transport_m3'] = df_contour['transport_dswt'].values * df_contour['dx'].values
    df_contour_daily = df_contour.groupby(['time']).agg(total_transport=('transport_m3', 'sum'))
    
    daily_transport = np.zeros(len(time))
    for t in range(len(df_contour_daily)):
        i_time = np.where(df_contour_daily.index[t] == time)[0][0]
        daily_transport[i_time] = df_contour_daily['total_transport'].values[t] / contour_length

    return time, daily_transport, depth_contour, contour_length

if __name__ == '__main__':
    input_dir = 'output/test_114-116E_33-31S/'
    years = [2017]
    
    time, f_dswt = read_dswt_occurrence_timeseries(input_dir, years)
    len(time)