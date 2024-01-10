import xarray as xr
import numpy as np
from datetime import datetime
import os
from tools.files import get_files_in_dir
from tools.roms import get_z, find_eta_xi_covering_lon_lat_box, convert_roms_u_v_to_u_east_v_north
from tools.roms import get_eta_xi_along_transect, get_distance_along_transect

def select_input_files(input_dir:str, file_contains=None,
                       remove_gridfile=True, filetype='nc') -> list[str]:
    all_files = get_files_in_dir(input_dir, filetype)
    if file_contains is not None:
        files = [f for f in all_files if file_contains in f]
    else:
        files = all_files
        
    if remove_gridfile is True:
        grid_files = [f for f in files if 'grid' in f]
        for f in grid_files:
            files.remove(f)
        
    return files

def load_roms_data(input_dir:str, grid_file=None, files_contain=None) -> xr.Dataset:
    
    paths = select_input_files(input_dir, file_contains=files_contain)
    roms_ds = xr.open_mfdataset(paths, data_vars='minimal')
    
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
        if 'hc' not in roms_ds.variables:
            roms_ds['hc'] = rg.hc
    
    # add layer depths z_rho    
    z_rho = get_z(roms_ds.Vtransform.values, roms_ds.s_rho.values, roms_ds.h.values, roms_ds.Cs_r.values, roms_ds.hc.values)
    roms_ds.coords['z_rho'] = (['s_rho', 'eta_rho', 'xi_rho'], z_rho)
    
    # convert u and v to u_east and v_north
    if 'u_east' not in roms_ds.variables:
        u_east, v_north = convert_roms_u_v_to_u_east_v_north(roms_ds.u.values, roms_ds.v.values, roms_ds.angle.values)
        roms_ds['u_east'] = (['ocean_time', 's_rho', 'eta_rho', 'xi_rho'], u_east)
        roms_ds['v_north'] = (['ocean_time', 's_rho', 'eta_rho', 'xi_rho'], v_north)

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

files = select_input_files('tests/data/', file_contains='cwa', remove_gridfile=False)
files = select_input_files('tests/data/', file_contains='cwa', remove_gridfile=True)
# roms_ds = load_roms_data('input/', grid_file='input/cwa_grid.nc')
# transect_ds = select_roms_transect(roms_ds, 115.7, -31.76, 115.26, -31.95)

# transect_ds.temp.sel(ocean_time='2017-06-13 03:00').plot(x='distance', y='z_rho')
