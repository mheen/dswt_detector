import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from readers.read_ocean_data import load_roms_data, select_roms_subset
from tools.config import Config, read_config
from tools.coordinates import get_bearing_between_points
from tools.files import get_dir_from_json
from tools.roms import get_eta_xi_of_lon_lat_point
from tools.roms import get_distance_along_transect as get_distance_between_points_array
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from warnings import warn

def get_angle_between_mask_and_velocity(mask:np.ndarray[bool],
                                        u:np.ndarray[float],
                                        v:np.ndarray[float]) -> np.ndarray[float]:
    '''Calculates the angle between a given mask (this can be a
    land mask or a mask indicating the continental shelf edge)
    and given velocities (current or wind). This is done by:
    
    The angle of the mask (land/shelf) can be determined by:
    alpha_mask = arctan2(dL/dx, dL/dy)
    since the gradient of the mask will indicate the direction.
    
    The dot product between two vectors is the cosine of the
    angle between vectors:
    cos(theta) = x.y/|x||y|
    
    So the angle between a mask and velocity can be calculated:
    theta = arccos((u dL/dx + v dL/dy)/(sqrt(u**2+v**2)sqrt(dL/dx**2+dL/dy**2)))'''

    dLdy, dLdx = np.gradient(mask)

    numerator = u*dLdx+v*dLdy
    denominator = np.sqrt(u**2+v**2)*np.sqrt(dLdx**2+dLdy**2)
    theta = np.arccos(numerator/denominator)

    return theta

def get_coordinates_of_depth_contour(roms_ds:xr.Dataset, h_contour=50) -> tuple[np.ndarray, np.ndarray]:
    ax = plt.axes()
    contour_set = ax.contour(roms_ds.lon_rho.values, roms_ds.lat_rho.values, roms_ds.h.values, levels=[h_contour])

    contour_line = contour_set.collections[0].get_paths()[0]
    contour_line_coords = contour_line.vertices
    lons = contour_line_coords[:, 0]
    lats = contour_line_coords[:, 1]

    plt.close()

    return lons, lats

def calculate_daily_mean_cross_shelf_transport_at_depth_contour(roms_ds:xr.Dataset,
                                                                h_contour:float,
                                                                s_layers:list[list[int]]) -> tuple:
    h_smooth = gaussian_filter(roms_ds.h.values, 2)
    theta = get_angle_between_mask_and_velocity(h_smooth, roms_ds.u_eastward.values, roms_ds.v_northward.values) # angle between bathymetry

    cross_shelf_vel = np.sqrt(roms_ds.u_eastward.values**2+roms_ds.v_northward.values**2)*np.cos(theta) # velocity perpendicular to bathymetry angle (downslope)
    
    # get cross-shelf transport along specific depth-contour only
    lons_contour, lats_contour = get_coordinates_of_depth_contour(roms_ds, h_contour=h_contour)
    eta, xi = get_eta_xi_of_lon_lat_point(roms_ds.lon_rho.values, roms_ds.lat_rho.values, lons_contour, lats_contour)
    distance = get_distance_between_points_array(lons_contour, lats_contour)
    ds = np.diff(distance)
    
    s_layers = s_layers.astype(int)

    cross_shelf_transport = np.nansum((cross_shelf_vel[:, s_layers, :, :][:, :, eta, xi]*roms_ds.delta_z.values[s_layers, :, :][:, eta, xi])[:, :, :-1]*ds*roms_ds.dt.values)
    mean_cross_shelf_vel = np.nanmean(cross_shelf_vel[:, s_layers, :, :][:, :, eta, xi])
    
    return cross_shelf_transport, mean_cross_shelf_vel

def calculate_down_transect_velocity_component(u:np.ndarray, v:np.ndarray,
                                               lon1:float, lat1:float,
                                               lon2:float, lat2:float) -> np.ndarray:
    alpha = get_bearing_between_points(lon1, lat1, lon2, lat2)
    alpha_rad = np.deg2rad(alpha)
    down_transect = u*np.sin(alpha_rad)+v*np.cos(alpha_rad)
    return down_transect

def add_down_transect_velocity_to_ds(transect_ds:xr.Dataset) -> xr.Dataset:
    down_transect_vel = calculate_down_transect_velocity_component(
            transect_ds.u_eastward.values,
            transect_ds.v_northward.values,
            transect_ds.lon_rho.values[0], # land location
            transect_ds.lat_rho.values[0],
            transect_ds.lon_rho.values[-1],
            transect_ds.lat_rho.values[-1])
    
    transect_ds['down_transect_vel'] = (['ocean_time', 's_rho', 'distance'], down_transect_vel)
    
    return transect_ds

def calculate_dswt_cross_shelf_transport_along_transect(
    transect_ds:xr.Dataset,
    l_dswt:np.array,
    config:Config) -> tuple[np.ndarray, np.ndarray]:

    if np.all(l_dswt == False):
        return [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan]
    
    transect_ds = add_down_transect_velocity_to_ds(transect_ds)

    transport = np.empty(len(transect_ds.distance))*np.nan
    mean_vel = np.empty(len(transect_ds.distance))*np.nan
    mean_dz = np.empty(len(transect_ds.distance))*np.nan
    sum_dz = np.empty(len(transect_ds.distance))*np.nan
    sum_vel = np.zeros(len(transect_ds.distance))
    vel_counts = np.zeros(len(transect_ds.distance))
    
    i_dswt = np.where(l_dswt == True)[0] # times where there is DSWT

    n_z_layers = len(transect_ds.s_rho)
    n_depth_layers = int(np.ceil(n_z_layers*config.drhodz_depth_percentage)) # depth layers to consider for DSWT
    for i in i_dswt:
        # locations along transect where drho/dz condition is met:
        i_dists = np.where(transect_ds.vertical_density_gradient[i, 0:n_depth_layers, :] >= config.minimum_drhodz)[1]
        if len(i_dists) == 0:
            continue # drho/dz condition not satisfied: should not really end up here since there should be DSWT somewhere in the transect
        
        i_dists = np.unique(i_dists)
        for j in i_dists:
            i_depth = np.where(transect_ds.vertical_density_gradient[i, 0:n_depth_layers, j] >= config.minimum_drhodz)[0]
            if len(i_depth) == 0:
                continue # drho/dz condition not satisfied at this location along shelf
            
            # calculate transport over all depth layers below first spike in drho/dz
            if i_depth[-1] == 0:
                k = 0
            else:
                k = i_depth[-1]-1
            # sum with already existing values at locations
            if np.all(np.isnan(transect_ds.down_transect_vel.values[i, 0:k, j])):
                # FIX! there seem to be some grid cells close to land that do not have a velocity but do have other parameters
                # this probably needs a fix in the conversion of u and v to rho points
                # I am skipping these cells for now, as these should be limited and only cells close to land
                continue
            if np.logical_and(i == 3, j == 8) == True:
                print('bla')
            transport[j] = np.nansum([transport[j], np.nansum(transect_ds.down_transect_vel.values[i, 0:k, j]*transect_ds.delta_z.values[0:k, j])*transect_ds.dt.values])
            depth_mean_vel = np.nansum(transect_ds.down_transect_vel.values[i, 0:k, j]*transect_ds.delta_z.values[0:k, j])/np.nansum(transect_ds.delta_z.values[0:k, j])
            sum_vel[j] = np.nansum([sum_vel[j], depth_mean_vel])
            vel_counts[j] += 1
            sum_dz[j] = np.nansum([sum_dz[j], np.nansum(transect_ds.delta_z.values[0:k, j])])
            
    mean_vel = sum_vel/vel_counts
    mean_dz = sum_dz/vel_counts
    
    l_nonzero = np.logical_and(np.logical_and(~np.isnan(transport), transport != 0.0),
                               np.logical_and(~np.isnan(mean_vel), mean_vel != 0.0))
    
    if np.sum(l_nonzero) == 0: # you shouldn't really end up here...
        warn(f'DSWT identified in transect, but no cross-shelf transport found, returning NaNs.')
        return [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan]
    
    return (transport[l_nonzero], mean_vel[l_nonzero], mean_dz[l_nonzero],
            transect_ds.distance.values[l_nonzero],
            transect_ds.lon_rho.values[l_nonzero], transect_ds.lat_rho.values[l_nonzero],
            transect_ds.h.values[l_nonzero])
