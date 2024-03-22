import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from readers.read_ocean_data import load_roms_data, select_roms_subset
from tools.config import Config, read_config
from tools.coordinates import get_bearing_between_points
from tools.files import get_dir_from_json
import numpy as np
import xarray as xr

def get_down_transect_velocity_component(u:np.ndarray, v:np.ndarray,
                                         lon1:float, lat1:float,
                                         lon2:float, lat2:float) -> np.ndarray:
    alpha = get_bearing_between_points(lon1, lat1, lon2, lat2)
    alpha_rad = np.deg2rad(alpha)
    down_transect = u*np.sin(alpha_rad)+v*np.cos(alpha_rad)
    return down_transect

def calculate_dswt_cross_shelf_transport_along_transect(
    transect_ds:xr.Dataset,
    transect:dict,
    l_dswt:np.array,
    config:Config) -> np.ndarray:

    if np.all(l_dswt) == False:
        return np.zeros(len(l_dswt))
    
    down_transect_v = get_down_transect_velocity_component(
            transect_ds.u_eastward.values,
            transect_ds.v_northward.values,
            transect['lon_land'], # potentially just use first lon and lat coordinate in transect_ds instead?
            transect['lat_land'],
            transect['lon_ocean'],
            transect['lat_ocean'])
    
    transport = np.zeros(len(transect_ds.ocean_time))
    i_dswt = np.where(l_dswt == True)[0] # times where there is DSWT
    i_dists = np.where(np.logical_and(transect_ds.h.values >= config.dswt_cross_shelf_transport_depth_range[0],
                            transect_ds.h.values <= config.dswt_cross_shelf_transport_depth_range[1]))[0] # locations for depth range
    n_z_layers = len(transect_ds.z_rho)
    n_depth_layers = int(np.ceil(n_z_layers*config.drhodz_depth_percentage)) # depth layers to consider for DSWT
    for i in i_dswt:
        # calculate transport over all depth layers below first spike in drho/dz
        transport_over_depth_range = []
        for j in i_dists:
            i_depth = np.where(transect_ds.vertical_density_gradient[i, 0:n_depth_layers, j] >= config.minimum_drhodz)[0]
            if len(i_depth) == 0:
                continue # drho/dz condition not satisfied at this location along shelf
            k = i_depth[-1]
            dz = np.diff(transect_ds.z_rho[0:k+1, j])
            transport_over_depth_range.append(np.nanmean(down_transect_v[i, 0:k, j]*dz)*transect['width']*transect_ds.dt.values) # m3
        
        if len(transport_over_depth_range) == 0:
            continue # drho/dz condition not satisfied over entire depth range: transport = 0
        
        transport[i] = np.nanmean(transport_over_depth_range)
    
    return transport
  
if __name__ == '__main__':
    transects_file = f'input/transects/cwa_transects.json'
    
    config = read_config('cwa')

    lon_range = [114.0, 116.0]
    lat_range = [-33.0, -31.0]

    roms_ds = load_roms_data(f'{get_dir_from_json("cwa")}2017/cwa_20170514_03__his.nc', f'{get_dir_from_json("cwa")}grid.nc')
    roms_ds = select_roms_subset(roms_ds, None, lon_range, lat_range)
