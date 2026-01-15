from tools.roms import get_eta_xi_of_lon_lat_point
from tools.config import Config

import pandas as pd
import numpy as np
import xarray as xr

def spatially_interpolate_dswt(df_transport:pd.DataFrame, interpolator, grid_ds:xr.Dataset, config:Config):
    columns = ['time', 'eta', 'xi', 'transport', 'mean_thickness', 'max_distance', 'min_distance', 'max_h', 'mean_drhodx']
    df_interp = pd.DataFrame(columns=columns)
    
    def _interpolate(x:np.ndarray, y:np.ndarray, z:np.ndarray, X:np.ndarray, Y:np.ndarray):
        interp = interpolator(list(zip(x, y)), z)
        Z = interp(X, Y)
        return Z.flatten()
    
    # interpolate to points with transect contour depths
    l_points = np.logical_and(grid_ds.h.values >= config.transect_contours[0], grid_ds.h.values <= config.transect_contours[-1])
    lon_interp = grid_ds.lon_rho.values[l_points]
    lat_interp = grid_ds.lat_rho.values[l_points]
    eta_interp, xi_interp = get_eta_xi_of_lon_lat_point(grid_ds.lon_rho.values, grid_ds.lat_rho.values, lon_interp.flatten(), lat_interp.flatten())
    
    times = np.unique(df_transport['time'].values)
    for t in times:
        l_time = df_transport['time'].values == t
        if np.sum(l_time) <= 4: # not enough points to interpolate
            continue
        eta = df_transport[l_time]['eta'].values
        xi = df_transport[l_time]['xi'].values
        x = grid_ds.lon_rho.values[eta.astype(int), xi.astype(int)]
        y = grid_ds.lat_rho.values[eta.astype(int), xi.astype(int)]
                
        transport = _interpolate(x, y, df_transport[l_time]['transport'].values, lon_interp, lat_interp)
        mean_thickness = _interpolate(x, y, df_transport[l_time]['mean_thickness'].values, lon_interp, lat_interp)
        max_distance = _interpolate(x, y, df_transport[l_time]['max_distance'].values, lon_interp, lat_interp)
        min_distance = _interpolate(x, y, df_transport[l_time]['min_distance'].values, lon_interp, lat_interp)
        max_h = _interpolate(x, y, df_transport[l_time]['max_h'].values, lon_interp, lat_interp)
        mean_drhodx = _interpolate(x, y, df_transport[l_time]['mean_drhodx'].values, lon_interp, lat_interp)
        l_nonan = ~np.isnan(transport)
        
        df_interp_t = pd.DataFrame(data=np.array([np.repeat(pd.to_datetime(t), sum(l_nonan.astype(int))),
                                                    eta_interp[l_nonan],
                                                    xi_interp[l_nonan],
                                                    transport[l_nonan],
                                                    mean_thickness[l_nonan],
                                                    max_distance[l_nonan],
                                                    min_distance[l_nonan],
                                                    max_h[l_nonan],
                                                    mean_drhodx[l_nonan]]).transpose(),
                                    columns=columns)
        
        df_interp = pd.concat([df_interp, df_interp_t])
        
    return df_interp