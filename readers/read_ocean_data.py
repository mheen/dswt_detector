import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

import xarray as xr
import numpy as np
from datetime import datetime
import os
from tools.files import get_files_in_dir
from tools import log
from tools.roms import get_z, find_eta_xi_covering_lon_lat_box, convert_roms_u_v_to_u_east_v_north
from tools.roms import get_eta_xi_along_transect, get_distance_along_transect
from tools.seawater_density import calculate_density
from tools.coordinates import get_bearing_between_points

g = 9.81 # m/s2

def find_filepath_for_specific_date(input_dir:str, date:datetime):
    input_dir_year = f'{input_dir}{date.strftime("%Y")}/'
    input_file = [p for p in os.listdir(input_dir_year) if date.strftime("%Y%m%d") in p][0]
    input_path = f'{input_dir_year}{input_file}'
    return input_path

def select_input_files(input_dir:str, file_preface=None,
                       date_range=None, dateformat='%Y%m%d',
                       remove_gridfile=True, filetype='nc', file_contains=None) -> list[str]:
    all_files = get_files_in_dir(input_dir, filetype, return_full_path=False)
    if file_preface is not None:
        files = [f for f in all_files if f.startswith(file_preface)]
    else:
        files = all_files
        
    if file_contains is not None:
        files = [f for f in files if file_contains in f]
        
    if remove_gridfile is True:
        grid_files = [f for f in files if 'grid' in f]
        for f in grid_files:
            files.remove(f)
            
    if date_range is not None:
        len_date = len(datetime(2100, 1, 1).strftime(dateformat))
        files = [f for f in files if date_range[0]<=datetime.strptime(f[:len(file_preface)+len_date], f'{file_preface}{dateformat}')<=date_range[1]]
    
    files = [f'{input_dir}{f}' for f in files]    
    return files

def read_roms_data(input_paths:str, grid_file:str, drop_vars:list) -> xr.Dataset:
    log.info(f'Reading ROMS data from files: {input_paths}')
    roms_ds = xr.open_mfdataset(input_paths, data_vars='minimal', drop_variables=drop_vars)
    
    # model dt
    dt = np.unique(np.diff(roms_ds.ocean_time).astype('timedelta64[s]').astype(float))[0]
    roms_ds['dt'] = dt
    
    # read grid variables if in separate file
    grid_variables = ['lon_rho', 'lat_rho', 'h', 'angle', 'Vtransform', 'Cs_r', 'Cs_w', 'hc', 'mask_rho']
    in_ds = [v in roms_ds.variables for v in grid_variables]
    if all(in_ds) is False:
        if grid_file is None:
            raise ValueError(f'No grid variables in ROMS files, expecting a separate grid file.')
        rg = xr.load_dataset(grid_file)
        
        if 'lon_rho' not in roms_ds.variables:
            roms_ds.coords['lon_rho'] = rg.lon_rho
        if 'lat_rho' not in roms_ds.variables:
            roms_ds.coords['lat_rho'] = rg.lat_rho
        if 'h' not in roms_ds.variables:
            roms_ds['h'] = rg.h
        if 'angle' not in roms_ds.variables:
            roms_ds['angle'] = rg.angle
        
        if 'Vtransform' not in roms_ds.variables:
            roms_ds['Vtransform'] = rg.Vtransform
        if 'Cs_r' not in roms_ds.variables:
            roms_ds['Cs_r'] = rg.Cs_r
        if 'Cs_w' not in roms_ds.variables:
            roms_ds['Cs_w'] = rg.Cs_w
        if 'hc' not in roms_ds.variables:
            roms_ds['hc'] = rg.hc
        if 'mask_rho' not in roms_ds.variables:
            roms_ds['mask_rho'] = rg.mask_rho
            
    return roms_ds

def convert_roms_u_and_v(roms_ds:xr.Dataset) -> xr.Dataset:
    # convert u and v to u_east and v_north
    if 'u_eastward' not in roms_ds.variables:
        u_eastward, v_northward = convert_roms_u_v_to_u_east_v_north(roms_ds.u.values, roms_ds.v.values, roms_ds.angle.values, roms_ds.mask_rho.values)
        roms_ds['u_eastward'] = (['ocean_time', 's_rho', 'eta_rho', 'xi_rho'], u_eastward)
        roms_ds['v_northward'] = (['ocean_time', 's_rho', 'eta_rho', 'xi_rho'], v_northward)
        
    return roms_ds
            
def add_variables_to_roms_data(roms_ds:xr.Dataset) -> xr.Dataset:
    # --- calculate layer depths z_rho and z_w
    z_rho = get_z(roms_ds.Vtransform.values, roms_ds.s_rho.values, roms_ds.h.values, roms_ds.Cs_r.values, roms_ds.hc.values)
    roms_ds.coords['z_rho'] = (['s_rho', 'eta_rho', 'xi_rho'], z_rho)
    
    z_w = get_z(roms_ds.Vtransform.values, roms_ds.s_w.values, roms_ds.h.values, roms_ds.Cs_w.values, roms_ds.hc.values)
    roms_ds.coords['z_w'] = (['s_w', 'eta_rho', 'xi_rho'], z_w)
    
    delta_z = np.diff(z_w, axis=0)
    roms_ds['delta_z'] = (['s_rho', 'eta_rho', 'xi_rho'], delta_z)
    
    # --- calculate seawater density if temperature and salinity available
    if 'salt' in roms_ds.variables and 'temp' in roms_ds.variables:
        density = calculate_density(roms_ds.salt.values, roms_ds.temp.values, -roms_ds.z_rho.values)
        roms_ds['density'] = (['ocean_time', 's_rho', 'eta_rho', 'xi_rho'], density)
    else:
        missing_variable = [v for v in ['salt', 'temp'] if v not in roms_ds.variables]
        log.info(f'Cannot calculate seawater density, missing ROMS variable: {missing_variable}')
    
    if 'density' in roms_ds.variables:
        # --- calculate depth mean density
        depth_mean_density = np.sum(roms_ds.density.values*roms_ds.delta_z.values, axis=1)/roms_ds.h.values
        roms_ds['depth_mean_density'] = (['ocean_time', 'eta_rho', 'xi_rho'], depth_mean_density)
        
        # --- calculate vertical density difference
        drho_z = np.diff(roms_ds.density.values, axis=1)
        # convert vertical density difference back to rho-points
        drho_z_rho = np.empty(roms_ds.density.shape) * np.nan
        drho_z_rho[:, 1:-1, :] = 0.5 * (drho_z[:, 0:-1, :] + drho_z[:, 1:, :])
        drho_z_rho[:, 0, :] = drho_z[:, 0, :]
        drho_z_rho[:, -1, :] = drho_z[:, -1, :]
        roms_ds['drho_z'] = (['ocean_time', 's_rho', 'eta_rho', 'xi_rho'], drho_z_rho)
        
        # --- calculate potential energy anomaly -> NOTE: possibly redundant? remove if so
        depth_mean_density_resized = np.repeat(depth_mean_density[:, np.newaxis, :, :], roms_ds.density.shape[1], axis=1)
        phi = g/roms_ds.h.values*np.sum((depth_mean_density_resized-roms_ds.density.values)*roms_ds.z_rho.values*roms_ds.delta_z.values, axis=1)
        roms_ds['potential_energy_anomaly'] = (['ocean_time', 'eta_rho', 'xi_rho'], phi)
        
    else:
        log.info(f'Cannot calculate depth mean density, vertical density gradient, potential energy anomaly: missing density variable.')
        
    return roms_ds

def calculate_down_transect_velocity_component(u:np.ndarray, v:np.ndarray,
                                               lon1:float, lat1:float,
                                               lon2:float, lat2:float) -> np.ndarray:
    alpha = get_bearing_between_points(lon1, lat1, lon2, lat2)
    alpha_rad = np.deg2rad(alpha)
    down_transect = u*np.sin(alpha_rad)+v*np.cos(alpha_rad)
    return down_transect

def add_variables_to_transect_data(transect_ds:xr.Dataset):
    # --- add distance along transect as a coordinate
    distance = get_distance_along_transect(transect_ds.lon_rho.values, transect_ds.lat_rho.values)
    transect_ds.coords['distance'] = distance
    
    dx = np.diff(distance)
    # convert dx back to rho-points
    dx_rho = np.empty(distance.shape) * np.nan
    dx_rho[1:-1] = 0.5 * (dx[0:-1] + dx[1:])
    dx_rho[0] = dx[0]
    dx_rho[-1] = dx[-1]
    transect_ds['delta_x'] = (['distance'], dx_rho)
    
    # --- add slope
    slope = np.diff(transect_ds.h) / dx
    # convert slope back to rho-points
    slope_rho = np.empty(distance.shape) * np.nan
    slope_rho[1:-1] = 0.5 * (slope[0:-1] + slope[1:])
    slope_rho[0] = slope[0]
    slope_rho[-1] = slope[-1]
    transect_ds['slope'] = (['distance'], slope_rho)
    
    # --- add horizontal density difference
    drho_x = np.diff(transect_ds.density.values, axis=2)
    drho_dx = drho_x / dx
    # convert drho_dx back to rho-points
    drho_dx_rho = np.empty(transect_ds.density.shape) * np.nan
    drho_dx_rho[:, :, 1:-1] = 0.5 * (drho_dx[:, :, 0:-1] + drho_dx[:, :, 1:])
    drho_dx_rho[:, :, 0] = drho_dx[:, :, 0]
    drho_dx_rho[:, :, -1] = drho_dx[:, :, -1]
    transect_ds['drho_dx'] = (['ocean_time', 's_rho', 'distance'], drho_dx_rho)
    
    # --- add depth mean horizontal density gradient
    drho_zmean = np.diff(transect_ds.depth_mean_density.values, axis=1)
    drho_dx_zmean = drho_zmean / dx
    # convert drho_dx_zmean back to rho-points
    drho_dx_zmean_rho = np.empty((len(transect_ds.ocean_time), len(transect_ds.distance))) * np.nan
    drho_dx_zmean_rho[:, 1:-1] = 0.5 * (drho_dx_zmean[:, 0:-1] + drho_dx_zmean[:, 1:])
    drho_dx_zmean_rho[:, 0] = drho_dx_zmean[:, 0]
    drho_dx_zmean_rho[:, -1] = drho_dx_zmean[:, -1]
    transect_ds['drho_dx_zmean'] = (['ocean_time', 'distance'], drho_dx_zmean_rho)
    
    # --- add down-transect velocity (positive down slope)
    u_down = calculate_down_transect_velocity_component(
                transect_ds.u_eastward.values,
                transect_ds.v_northward.values,
                transect_ds.lon_rho.values[0], # land location
                transect_ds.lat_rho.values[0],
                transect_ds.lon_rho.values[-1],
                transect_ds.lat_rho.values[-1])
    
    transect_ds['u_down'] = (['ocean_time', 's_rho', 'distance'], u_down)
    
    return transect_ds

def load_roms_data(input_path:str, grid_file=None, drop_vars=None) -> xr.Dataset:
    if not type(input_path) == list:
        input_path = [input_path]
    roms_ds = read_roms_data(input_path, grid_file, drop_vars=drop_vars)
    roms_ds = convert_roms_u_and_v(roms_ds)
    roms_ds = add_variables_to_roms_data(roms_ds)
    
    return roms_ds

def load_mf_roms_data(input_dir:str, grid_file=None, files_contain=None, drop_vars=None) -> xr.Dataset:
    input_paths = select_input_files(input_dir, file_contains=files_contain)
    roms_ds = read_roms_data(input_paths, grid_file, drop_vars)
    roms_ds = convert_roms_u_and_v(roms_ds)
    roms_ds = add_variables_to_roms_data(roms_ds)

    return roms_ds

def select_roms_subset(roms_ds:xr.Dataset,
                       time_range:list,
                       lon_range:list,
                       lat_range:list) -> xr.Dataset:
    
    if isinstance(time_range, list):
        subset_ds = roms_ds.sel(ocean_time=slice(time_range[0], time_range[1]))
    elif isinstance(time_range, str):
        subset_ds = roms_ds.sel(ocean_time=time_range)
    else:
        subset_ds = roms_ds
    
    if lon_range is not None and lat_range is not None:
        xi0, xi1, eta0, eta1 = find_eta_xi_covering_lon_lat_box(roms_ds.lon_rho.values, roms_ds.lat_rho.values, lon_range, lat_range)
        subset_ds = subset_ds.sel(xi_rho=slice(xi0, xi1), eta_rho=slice(eta0, eta1))
    
    return subset_ds

def select_roms_transect_from_known_coordinates(roms_ds:xr.Dataset, eta:np.ndarray, xi:np.ndarray) -> xr.Dataset:
    etas = xr.DataArray(eta, dims='distance') # conversion to xr.DataArray needed to select individual points (rather than grid)
    xis = xr.DataArray(xi, dims='distance') # naming dimension "distance" here allows coordinate values to be linked to it later
   
    transect_ds = roms_ds.sel(xi_rho=xis, eta_rho=etas)
    transect_ds = add_variables_to_transect_data(transect_ds)
    
    return transect_ds

def select_roms_transect_from_start_end_coordinates(
    roms_ds:xr.Dataset,
    lon1:float, lat1:float,
    lon2:float, lat2:float,
    ds=500.
    ) -> xr.Dataset:
    eta, xi = get_eta_xi_along_transect(roms_ds.lon_rho.values, roms_ds.lat_rho.values, lon1, lat1, lon2, lat2, ds)
    etas = xr.DataArray(eta, dims='distance') # conversion to xr.DataArray needed to select individual points (rather than grid)
    xis = xr.DataArray(xi, dims='distance') # naming dimension "distance" here allows coordinate values to be linked to it later
    
    transect_ds = roms_ds.sel(xi_rho=xis, eta_rho=etas)
    transect_ds = add_variables_to_transect_data(transect_ds)
    
    return transect_ds
