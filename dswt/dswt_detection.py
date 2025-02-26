import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.files import get_dir_from_json
from tools.config import read_config
from transects import read_transects_in_lon_lat_range_from_json
from readers.read_ocean_data import load_roms_data

from readers.read_ocean_data import select_roms_transect_from_known_coordinates
from dswt.cross_shelf_transport import calculate_dswt_cross_shelf_transport_along_transect
from tools.config import Config
from itertools import groupby
import numpy as np
import xarray as xr
import pandas as pd

def calculate_horizontal_density_gradient_along_transect(transect_ds:xr.Dataset) -> float:
    drho = np.diff(transect_ds.depth_mean_density.values, axis=1)
    dx = np.diff(transect_ds.distance.values)
    
    drhodx = drho/dx
    
    # add dummy value at end for size consistency
    # !!! FIX !!! find a better way to do this
    drhodx = np.hstack((drhodx, np.expand_dims(drhodx[:, -1], 1)))

    return drhodx

def determine_dswt_along_transect(transect_ds:xr.Dataset, config:Config):
    '''Determines if DSWT is occurring along a transect or not.
    Conditions for DSWT are:
    1. There must be a negative horizontal density gradient along the transect
       (higher depth mean density at the coast, decreasing density towards open ocean).
       This is determined by taking the mean horizontal density gradient over the entire
       transect, and so returns one bool per time.
       The default value for minimum_drhodz = 0.02, but this needs to be chosed with care
       depending on the ocean model used. The "plot_cycling_..." functions in this script
       can be used to help determine what this value should be.
    2. The vertical density gradient must exceed a minimum value (minimum_drhodz) for
       at least a minimum amount of consecutive grid cells (determined as a percentage
       of the total number of cells along the transect: minimum_p_cells). This must happen
       in the bottom layers of the model, specified as a percentage of cells (drhodz_depth_p).
       This condition also returns one bool per time over the entire transect.
       The default value for minimum_p_cells = 0.3 (30%), which should hopefully work
       independent of the ocean model used.
       The default value for drhodz_depth_p = 0.5 (50%), which should hopefully work
       independent of the ocean model used. This condition was added because there
       can sometimes be high drhodz values at the ocean surface (for example when
       there are high salinity values at the ocean surface and when temperature values
       do not change much).
    
    The depth up to which data along the transect is used can be filtered by setting a maximum
    depth value for filter_depth. The default value for filter_depth = 100.0 (m). This filter
    option can be useful to prevent colder deep water (off the shelf) from showing up as DSWT.
    This option may be redundant with the drho/dz condition used though.'''
    
    # remove data for depths above filter_depth_up_to
    if config.filter_depth is not None:
        # mask (replace with NaNs) all values where depth > filter_depth:
        transect_ds = transect_ds.where(transect_ds.h < config.filter_depth)
    
    # condition 1: horizontal density gradient (away from coast) must be negative
    # this is determined per time over the entire transect
    drhodx = calculate_horizontal_density_gradient_along_transect(transect_ds)
    condition1 = np.nanmean(drhodx, axis=1) < 0. # taking mean over entire transect
    
    # condition 2: vertical density gradient > a minimum value (differs per ocean model)
    # AND the vertical density gradient must exceed this value for a minimum number of cells
    # the result is that DSWT is determined for the entire transect
    n_z_layers = len(transect_ds.s_rho)
    n_depth_layers = int(np.ceil(n_z_layers*config.drhodz_depth_percentage))
    l_drhodz = np.any(transect_ds.vertical_density_gradient[:, 0:n_depth_layers, :] > config.minimum_drhodz, axis=1) # [time, distance] (check for any along depth)
    n_used_cells = sum(~np.isnan(transect_ds.h.values))
    drhodz_max = []
    drhodz_cells = []
    condition2 = []
    for t in range(len(transect_ds.ocean_time)):
        consecutive_true_lengths = [len(list(v)) for k, v in groupby(l_drhodz.values[t, :]) if k == True]
        if len(consecutive_true_lengths) != 0:
            n_consecutive_cells = max(consecutive_true_lengths)
        else:
            n_consecutive_cells = 0
        drhodz_cells.append(n_consecutive_cells/n_used_cells)
        drhodz_max.append(np.nanmax(transect_ds.vertical_density_gradient[t, 0:n_depth_layers, :]))
        condition2.append((n_consecutive_cells/n_used_cells) >= config.minimum_percentage_consecutive_cells)
    condition2 = np.array(condition2)
    
    # DSWT along a transect when both condition 1 and condition 2 hold
    l_dswt = np.logical_and(condition1, condition2)
    
    return l_dswt, condition1, condition2, drhodz_max, drhodz_cells

def determine_daily_dswt_along_multiple_transects(roms_ds:xr.Dataset, transects:dict, config:Config) -> pd.DataFrame:
    
    transect_names = list(transects.keys())
    
    df_transects_dswt = pd.DataFrame(index=np.arange(0, len(transect_names)),
                                     columns=['time', 'transect', 'f_dswt', 'vel_dswt', 'transport_dswt', 'ds', 'dz_dswt', 'lon_transport', 'lat_transport', 'depth_transport'])
    time = pd.to_datetime(roms_ds.ocean_time.values[0]).date()
    row = 0
    for i, transect_name in enumerate(transect_names):
        eta = transects[transect_name]['eta']
        xi = transects[transect_name]['xi']
        
        transect_ds = select_roms_transect_from_known_coordinates(roms_ds, eta, xi)
        l_dswt, _, _, _, _ = determine_dswt_along_transect(transect_ds, config)
        
        dswt_cross_transport, dswt_cross_vel, dswt_dz, ds, lon, lat, h = calculate_dswt_cross_shelf_transport_along_transect(transect_ds, l_dswt, config)
        
        for j in range(len(dswt_cross_transport)):
            df_transects_dswt.loc[row] = [time, transect_name, np.nanmean(l_dswt.astype(int)),
                                          dswt_cross_vel[j], dswt_cross_transport[j], ds[j], dswt_dz[j], lon[j], lat[j], h[j]]
            row += 1
        
    return df_transects_dswt

if __name__ == '__main__':
    output_dswt = 'output/cwa_114-116E_33-31S/dswt_2017.csv'
    
    model_input_dir = get_dir_from_json('cwa')
    files = ['cwa_20170420_03__his.nc']
    grid_file = f'{model_input_dir}grid.nc'
    
    lon_range = [114., 116.]
    lat_range = [-33., -31.]
    transects = read_transects_in_lon_lat_range_from_json('input/transects/cwa_transects.json', lon_range, lat_range)
    
    config = read_config('cwa')
    
    for i in range(len(files)):
        roms_ds = load_roms_data(f'{model_input_dir}2017/{files[i]}', grid_file=grid_file)
        
        df_transects_dswt = determine_daily_dswt_along_multiple_transects(roms_ds, transects, config)
        if os.path.exists(output_dswt):
            df_transects_dswt.to_csv(output_dswt, mode='a', header=False, index=False)
        else:
            df_transects_dswt.to_csv(output_dswt, index=False)