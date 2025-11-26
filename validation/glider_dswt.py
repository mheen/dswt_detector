import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from readers.read_ocean_data import calculate_down_transect_velocity_component
from tools.config import read_config, Config
from tools.files import get_dir_from_json, get_files_in_dir
import numpy as np
import xarray as xr

RHO0 = 1025.

def load_glider_transect_data(input_path:str):
    transect_ds = xr.load_dataset(input_path)
    
    l_below_seafloor = transect_ds.z <= transect_ds.h
    transect_ds.density.values[l_below_seafloor] = np.nan
    transect_ds.temp.values[l_below_seafloor] = np.nan
    transect_ds.salt.values[l_below_seafloor] = np.nan
    transect_ds.u.values[l_below_seafloor] = np.nan
    transect_ds.v.values[l_below_seafloor] = np.nan
    
    dx = np.diff(transect_ds.distance.values)
    # convert dx back to rho-points
    dx_rho = np.empty(transect_ds.distance.shape) * np.nan
    dx_rho[1:-1] = 0.5 * (dx[0:-1] + dx[1:])
    dx_rho[0] = dx[0]
    dx_rho[-1] = dx[-1]
    transect_ds['delta_x'] = (['distance'], dx_rho)
    
    # --- add slope
    slope = np.diff(abs(transect_ds.h.values)) / dx
    # convert slope back to rho-points
    slope_rho = np.empty(transect_ds.distance.shape) * np.nan
    slope_rho[1:-1] = 0.5 * (slope[0:-1] + slope[1:])
    slope_rho[0] = slope[0]
    slope_rho[-1] = slope[-1]
    transect_ds['slope'] = (['distance'], slope_rho)
    
    # --- add depth mean density
    delta_z = np.diff(transect_ds.z)
    delta_z_rho = np.empty(transect_ds.z.shape) * np.nan
    delta_z_rho[1:-1] = 0.5 * (delta_z[0:-1] + delta_z[1:])
    delta_z_rho[0] = delta_z[0]
    delta_z_rho[-1] = delta_z[-1]
    delta_z_rho_2d = np.repeat(np.expand_dims(delta_z_rho, 1), len(transect_ds.distance), axis=1)
    delta_z_rho_2d[l_below_seafloor] = np.nan
    transect_ds['delta_z'] = (['z', 'distance'], delta_z_rho_2d)
    
    depth_mean_density = np.nansum(transect_ds.density.values * transect_ds.delta_z.values, axis=0) / np.nansum(transect_ds.delta_z.values, axis=0)
    transect_ds['depth_mean_density'] = (['distance'], depth_mean_density)
    
    # -- add vertical density difference
    drho_z = np.diff(transect_ds.density.values, axis=0)
    # convert vertical density difference back to rho-points
    drho_z_rho = np.empty(transect_ds.density.shape) * np.nan
    drho_z_rho[1:-1, :] = 0.5 * (drho_z[0:-1, :] + drho_z[1:, :])
    drho_z_rho[0, :] = drho_z[0, :]
    drho_z_rho[-1, :] = drho_z[-1, :]
    transect_ds['drho_z'] = (['z', 'distance'], drho_z_rho)
    
    # --- add horizontal density difference
    drho_x = np.diff(transect_ds.density.values, axis=1)
    drho_dx = drho_x / dx
    # convert drho_dx back to rho-points
    drho_dx_rho = np.empty(transect_ds.density.shape) * np.nan
    drho_dx_rho[:, 1:-1] = 0.5 * (drho_dx[:, 0:-1] + drho_dx[:, 1:])
    drho_dx_rho[:, 0] = drho_dx[:, 0]
    drho_dx_rho[:, -1] = drho_dx[:, -1]
    transect_ds['drho_dx'] = (['z', 'distance'], drho_dx_rho)
    
    # --- add depth mean horizontal density gradient
    drho_zmean = np.diff(transect_ds.depth_mean_density.values, axis=0)
    drho_dx_zmean = drho_zmean / dx
    # convert drho_dx_zmean back to rho-points
    drho_dx_zmean_rho = np.empty(transect_ds.distance.shape) * np.nan
    drho_dx_zmean_rho[1:-1] = 0.5 * (drho_dx_zmean[0:-1] + drho_dx_zmean[1:])
    drho_dx_zmean_rho[0] = drho_dx_zmean[0]
    drho_dx_zmean_rho[-1] = drho_dx_zmean[-1]
    transect_ds['drho_dx_zmean'] = (['distance'], drho_dx_zmean_rho)
    
    # --- add down-transect velocity (positive down slope)
    u_down = calculate_down_transect_velocity_component(
                transect_ds.u.values,
                transect_ds.v.values,
                transect_ds.lon.values[0], # land location
                transect_ds.lat.values[0],
                transect_ds.lon.values[-1],
                transect_ds.lat.values[-1])
    
    transect_ds['u_down'] = (['z', 'distance'], u_down)
    
    return transect_ds

def determine_dswt_along_glider_transect(transect_ds:xr.Dataset, config:Config):
    '''Conditions:
    1. Mean drho/dx < 0
    2. drho * s / rho0 < -2 10**-8
    3. transport in the bottom layer must be offshore (if available in glider data)'''
    
    # remove data for depths above filter_depth_up_to
    if config.filter_depth is not None:
        # replace with NaNs all values where depth > filter_depth:
        l_depth = abs(transect_ds.h.values) < config.filter_depth
        # transect_ds.slope.values[~l_depth] = np.nan
        transect_ds.drho_z.values[:, ~l_depth] = np.nan
        transect_ds.drho_dx_zmean.values[~l_depth] = np.nan
        transect_ds.u_down.values[:, ~l_depth] = np.nan
    
    # condition 1: mean depth mean horizontal density gradient (away from coast) must be negative
    mean_drhodx = np.nanmean(transect_ds.drho_dx_zmean.values)
    drhodx_condition = mean_drhodx < 0. # drhodx_condition: [ocean_time]
    
    if drhodx_condition == True:
        # condition 2: vertical density gradient needs to be sufficiently large
        drho_s = (transect_ds.drho_z.values * transect_ds.slope.values) / RHO0
        drho_s_condition = drho_s < -2*10**-8 # drho_s_condition: [ocean_time, s_rho, distance]
        # consider only vertical density gradient in bottom layers (remove any True values from surface layers)
        minimum_depth = transect_ds.h.values + config.drhodz_depth_percentage * abs(transect_ds.h.values)
        z = np.repeat(np.expand_dims(transect_ds.z.values, 1), len(transect_ds.distance), axis=1)
        l_shallow = z > minimum_depth
        drho_s_condition[l_shallow] = False
        
        l_dswt = drho_s_condition
        
        if np.all(np.isnan(transect_ds.u_down.values)) == False:
            all_transport = transect_ds.u_down.values * transect_ds.delta_z.values
        
        transport_dswt = np.zeros(len(transect_ds.distance)) * np.nan
        depth_mean_vel_dswt = np.zeros(len(transect_ds.distance)) * np.nan
        thickness_dswt = np.zeros(len(transect_ds.distance))
        min_drho_s = np.zeros(len(transect_ds.distance))
        mean_drho_s = np.zeros(len(transect_ds.distance))
        
        x_dswt = np.where(np.any(l_dswt, axis=0))[0]
        for x in x_dswt:
            z_dswt = np.where(l_dswt[:, x] == True)[0][-1] # shallowest layer up to which DSWT extends
            thickness_dswt[x] = np.nansum(transect_ds.delta_z.values[0:z_dswt + 1, x])
            min_drho_s[x] = np.nanmin(drho_s[0:z_dswt + 1, x])
            mean_drho_s[x] = np.nanmean(drho_s[0:z_dswt + 1, x])
            
            # calculate transport from bottom up to shallowest layer where there is DSWT
            if np.all(np.isnan(transect_ds.u_down.values[:, x])) == False:
                u_down_condition = transect_ds.u_down.values[:, x] > 0.
                transport_dswt[x] = np.nansum(all_transport[0:z_dswt + 1, x] * u_down_condition[0:z_dswt + 1].astype(int))
                depth_mean_vel_dswt[x] = transport_dswt[x] / thickness_dswt[x]
        
        return (depth_mean_vel_dswt[x_dswt], thickness_dswt[x_dswt], transport_dswt[x_dswt],
                transect_ds.distance.values[x_dswt], transect_ds.lon.values[x_dswt], transect_ds.lat.values[x_dswt],
                abs(transect_ds.h.values[x_dswt]), mean_drhodx, mean_drho_s[x_dswt], min_drho_s[x_dswt])
        
    return (np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]),
            np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), mean_drhodx,
            np.array([np.nan]), np.array([np.nan]))


if __name__ == '__main__':
    input_dir = get_dir_from_json('glider_transects')
    
    transect_files = get_files_in_dir(input_dir, 'nc')
    
    transect_ds = load_glider_transect_data(transect_files[0])
    config = read_config('test')
    
    vel, thickness, transport, distance, lon, lat, h, drhodx, mean_drhos, min_drhos = determine_dswt_along_glider_transect(transect_ds, config)
