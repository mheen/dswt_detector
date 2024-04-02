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

g = 9.81 # m/s2

def select_input_files(input_dir:str, file_preface=None,
                       date_range=None, dateformat='%Y%m%d',
                       remove_gridfile=True, filetype='nc') -> list[str]:
    all_files = get_files_in_dir(input_dir, filetype, return_full_path=False)
    if file_preface is not None:
        files = [f for f in all_files if f.startswith(file_preface)]
    else:
        files = all_files
        
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
    if 'lon_rho' not in roms_ds.variables:
        if grid_file is None:
            raise ValueError(f'No grid variables in ROMS files, expecting a separate grid file.')
        rg = xr.load_dataset(grid_file)
        roms_ds.coords['lon_rho'] = rg.lon_rho
        roms_ds.coords['lat_rho'] = rg.lat_rho
        roms_ds['h'] = rg.h
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
        u_eastward, v_northward = convert_roms_u_v_to_u_east_v_north(roms_ds.u.values, roms_ds.v.values, roms_ds.angle.values)
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
        
        # --- calculate vertical density gradient
        delta_z_rho = np.diff(roms_ds.z_rho.values, axis=0)
        drhodz = -np.diff(roms_ds.density, axis=1)/delta_z_rho # minus diff(density) because 0 element is at bottom
        # add dummy drhodz at surface so same number of s_rho-layers
        # !!! FIX !!! better way to do this?
        drhodz_resized = np.hstack((drhodz, np.expand_dims(drhodz[:, -1, :, :], 1))) # duplicating values along the surface
        roms_ds['vertical_density_gradient'] = (['ocean_time', 's_rho', 'eta_rho', 'xi_rho'], drhodz_resized)
        
        # --- calculate potential energy anomaly
        depth_mean_density_resized = np.repeat(depth_mean_density[:, np.newaxis, :, :], roms_ds.density.shape[1], axis=1)
        phi = g/roms_ds.h.values*np.sum((depth_mean_density_resized-roms_ds.density.values)*roms_ds.z_rho.values*roms_ds.delta_z.values, axis=1)
        roms_ds['potential_energy_anomaly'] = (['ocean_time', 'eta_rho', 'xi_rho'], phi)
        
    else:
        log.info(f'Cannot calculate depth mean density, vertical density gradient, potential energy anomaly: missing density variable.')
        
    return roms_ds

def load_roms_data(input_path:str, grid_file=None, drop_vars=None) -> xr.Dataset:
    roms_ds = read_roms_data([input_path], grid_file, drop_vars=drop_vars)
    roms_ds = convert_roms_u_and_v(roms_ds)
    roms_ds = add_variables_to_roms_data(roms_ds)
    
    return roms_ds

def load_mf_roms_data(input_dir:str, grid_file=None, files_contain=None, drop_vars=None) -> xr.Dataset:
    input_paths = select_input_files(input_dir, file_contains=files_contain)
    roms_ds = read_roms_data(input_paths, grid_file, files_contain, drop_vars=drop_vars)
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

def select_roms_transect(roms_ds:xr.Dataset,
                         lon1:float, lat1:float,
                         lon2:float, lat2:float,
                         ds=500.) -> xr.Dataset:
    eta, xi = get_eta_xi_along_transect(roms_ds.lon_rho.values, roms_ds.lat_rho.values, lon1, lat1, lon2, lat2, ds)
    etas = xr.DataArray(eta, dims='distance') # conversion to xr.DataArray needed to select individual points (rather than grid)
    xis = xr.DataArray(xi, dims='distance') # naming dimension "distance" here allows coordinate values to be linked to it later
    
    transect_ds = roms_ds.sel(xi_rho=xis, eta_rho=etas)
    
    distance = get_distance_along_transect(transect_ds.lon_rho.values, transect_ds.lat_rho.values)
    transect_ds.coords['distance'] = distance
    
    return transect_ds
