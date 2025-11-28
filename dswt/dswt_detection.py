import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.files import get_dir_from_json
from tools.config import read_config, Config
from transects import read_transects_in_lon_lat_range_from_json
from readers.read_ocean_data import load_roms_data

from readers.read_ocean_data import select_roms_transect_from_known_coordinates
from itertools import groupby
import numpy as np
import xarray as xr
import pandas as pd

RHO0 = 1025.

def determine_dswt_along_transect(transect_ds:xr.Dataset, config:Config, mld_condition=True):
    '''New method:
    1. MLD at coast must reach bottom (NOTE: think about how to implement this for island transects)
    2. drho/dx < 0
    3. drho * s / rho0 < -2 10**-8
    4. transport in the bottom layer must be offshore'''
    
    # remove data for depths above filter_depth_up_to
    if config.filter_depth is not None:
        # replace with NaNs all values where depth > filter_depth:
        l_depth = transect_ds.h.values < config.filter_depth
        # transect_ds.slope.values[~l_depth] = np.nan
        transect_ds.drho_z.values[:, :, ~l_depth] = np.nan
        transect_ds.drho_dx_zmean.values[:, ~l_depth] = np.nan
        transect_ds.u_down.values[:, :, ~l_depth] = np.nan
    
    # condition 1: mixed layer depth must reach the seafloor at the coast
    if mld_condition == True:
        i_coast = np.where(~np.isnan(transect_ds.density.values))[2][0]
        drho = transect_ds.density[:, -1, i_coast] - transect_ds.density[:, 0, i_coast] # s_rho[0] = bottom, s_rho[-1] = surface
        mld_condition = abs(drho) / RHO0 < 10**-4 # mld_condition : [ocean_time]
    else:
        mld_condition = np.ones(len(transect_ds.ocean_time)).astype(bool)
    
    # condition 2: mean depth mean horizontal density gradient (away from coast) must be negative
    mean_drhodx = np.nanmean(transect_ds.drho_dx_zmean.values, axis=1)
    drhodx_condition = mean_drhodx < 0. # drhodx_condition: [ocean_time]
    
    l_time_dswt_possible = np.logical_and(mld_condition, drhodx_condition)
    
    if np.any(l_time_dswt_possible) == True:
        # condition 3: vertical density gradient needs to be sufficiently large
        drho_s = (transect_ds.drho_z.values * transect_ds.slope.values) / RHO0
        drho_s_condition = drho_s < -2*10**-8 # drho_s_condition: [ocean_time, s_rho, distance]
        # consider only vertical density gradient at certain depth (remove any True values from surface layers)
        minimum_depth = transect_ds.h.values - config.drhodz_depth_percentage * transect_ds.h.values # note: h is positive
        l_shallow = transect_ds.z_rho.values > -minimum_depth # note: z_rho is negative, minimum depth is positive
        drho_s_condition[:, l_shallow] = False
        
        # condition 4: down slope velocity needs to be positive (offshore/down-slope)
        u_down_condition = transect_ds.u_down.values > 0. # u_down_condition: [ocean_time, s_rho, distance]
        
        l_dswt = np.logical_and(drho_s_condition[l_time_dswt_possible, :, :], u_down_condition[l_time_dswt_possible, :, :])
        
        all_transport = transect_ds.u_down.values * transect_ds.delta_z.values * transect_ds.dt.values
        
        t_dswt = np.unique(np.where(l_dswt == True)[0])
        if len(t_dswt) == 0:
            return (np.array([0]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]),
            np.array([np.nan]), np.nan, np.nan, np.array([np.nan]), np.array([np.nan]))
        
        transport_dswt = np.zeros((len(transect_ds.ocean_time), len(transect_ds.distance)))
        depth_mean_vel_dswt = np.empty((len(transect_ds.ocean_time), len(transect_ds.distance))) * np.nan
        thickness_dswt = np.empty((len(transect_ds.ocean_time), len(transect_ds.distance))) * np.nan
        min_drho_s = np.empty((len(transect_ds.ocean_time), len(transect_ds.distance))) * np.nan
        mean_drho_s = np.empty((len(transect_ds.ocean_time), len(transect_ds.distance))) * np.nan
        f_dswt = np.zeros(len(transect_ds.distance))
        for t in t_dswt:
            x_dswt = np.where(np.any(l_dswt[t, :, :], axis=0))[0]
            for x in x_dswt:
                f_dswt[x] += 1
                
                z_dswt = np.where(l_dswt[t, :, x] == True)[0][-1] # shallowest layer up to which DSWT extends
                # calculate transport from bottom up to shallowest layer where there is DSWT
                transport_dswt[t, x] = np.nansum(all_transport[t, 0:z_dswt + 1, x] * u_down_condition[t, 0:z_dswt + 1, x].astype(int))
                thickness_dswt[t, x] = np.nansum(transect_ds.delta_z.values[0:z_dswt + 1, x])
                depth_mean_vel_dswt[t, x] = transport_dswt[t, x] / (transect_ds.dt.values * thickness_dswt[t, x])
                min_drho_s[t, x] = np.nanmin(drho_s[t, 0:z_dswt + 1, x] * u_down_condition[t, 0:z_dswt + 1, x].astype(int))
                mean_drho_s[t, x] = np.nanmean(drho_s[t, 0:z_dswt + 1, x] * u_down_condition[t, 0:z_dswt + 1, x].astype(int))
                
        daily_transport_dswt = np.nansum(transport_dswt, axis=0)
        daily_mean_thickness_dswt = np.nanmean(thickness_dswt, axis=0)
        daily_mean_vel_dswt = np.nanmean(depth_mean_vel_dswt, axis=0)
        daily_min_drho_s = np.nanmin(min_drho_s, axis=0)
        daily_mean_drho_s = np.nanmean(mean_drho_s, axis=0)
        daily_mean_drhodx = np.nanmean(mean_drhodx[t_dswt])
        daily_min_drhodx = np.nanmin(mean_drhodx[t_dswt])
        f_dswt = f_dswt / len(transect_ds.ocean_time)
        
        x_dswt_all = np.where(daily_transport_dswt != 0)[0]
        
        return (f_dswt[x_dswt_all], daily_mean_vel_dswt[x_dswt_all], daily_mean_thickness_dswt[x_dswt_all],
                daily_transport_dswt[x_dswt_all], transect_ds.distance.values[x_dswt_all],
                transect_ds.lon_rho.values[x_dswt_all], transect_ds.lat_rho.values[x_dswt_all],
                transect_ds.h.values[x_dswt_all], daily_mean_drhodx, daily_min_drhodx,
                daily_mean_drho_s[x_dswt_all], daily_min_drho_s[x_dswt_all])
        
    return (np.array([0]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]),
            np.array([np.nan]), np.nan, np.nan, np.array([np.nan]), np.array([np.nan]))


def determine_daily_dswt_along_multiple_transects(roms_ds:xr.Dataset, transects:dict, config:Config) -> pd.DataFrame:
    
    transect_names = list(transects.keys())
    
    df_transects_dswt = pd.DataFrame(index=np.arange(0, len(transect_names)),
                                     columns=['time', 'transect',
                                              'f_dswt', 'vel', 'thickness', 'transport',
                                              'distance', 'lon', 'lat', 'h',
                                              'drhodx_mean', 'drhodx_min', 'drhos_mean', 'drhos_min'])
    time = pd.to_datetime(roms_ds.ocean_time.values[0]).date()
    row = 0
    for i, transect_name in enumerate(transect_names):
        eta = transects[transect_name]['eta']
        xi = transects[transect_name]['xi']
        
        transect_ds = select_roms_transect_from_known_coordinates(roms_ds, eta, xi)
        (f_dswt, vel, thickness, transport, distance, lon, lat, h, drhodx_mean, drhodx_min, drhos_mean, drhos_min) = determine_dswt_along_transect(transect_ds, config)
        
        for j in range(len(transport)):
            df_transects_dswt.loc[row] = [time, transect_name, f_dswt[j],
                                        vel[j], thickness[j], transport[j],
                                        distance[j], lon[j], lat[j], h[j],
                                        drhodx_mean, drhodx_min, drhos_mean[j], drhos_min[j]]
            row += 1
        
    return df_transects_dswt

if __name__ == '__main__':
    output_dswt = 'output/test_114-116E_33-31S/dswt_2017.csv'
    
    model_input_dir = get_dir_from_json('test_data')
    files = ['test_20170601_03__his.nc']
    grid_file = f'{model_input_dir}grid.nc'
    
    lon_range = [114., 116.]
    lat_range = [-33., -31.]
    transects = read_transects_in_lon_lat_range_from_json('input/transects/test_transects.json', lon_range, lat_range)
    
    config = read_config('test')
    
    for i in range(len(files)):
        roms_ds = load_roms_data(f'{model_input_dir}2017/{files[i]}', grid_file=grid_file)
        
        df_transects_dswt = determine_daily_dswt_along_multiple_transects(roms_ds, transects, config)
        if os.path.exists(output_dswt):
            df_transects_dswt.to_csv(output_dswt, mode='a', header=False, index=False)
        else:
            df_transects_dswt.to_csv(output_dswt, index=False)