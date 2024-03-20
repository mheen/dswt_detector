import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from read_ocean_data import load_roms_data, select_roms_transect
from itertools import groupby
import numpy as np
import xarray as xr
import json

minimum_drhodz_default = 0.02
minimum_p_cells_default = 0.1
drhodz_depth_p_default = 0.5
filter_depth_default = 100.

def calculate_horizontal_density_gradient_along_transect(transect_ds:xr.Dataset) -> float:
    drho = np.diff(transect_ds.depth_mean_density.values, axis=1)
    dx = np.diff(transect_ds.distance.values)
    
    drhodx = drho/dx
    
    # add dummy value at end for size consistency
    # !!! FIX !!! find a better way to do this
    drhodx = np.hstack((drhodx, np.expand_dims(drhodx[:, -1], 1)))

    return drhodx

def determine_dswt_along_transect(transect_ds:xr.Dataset,
                                  minimum_drhodz=minimum_drhodz_default,
                                  minimum_p_cells=minimum_p_cells_default,
                                  drhodz_depth_p=drhodz_depth_p_default,
                                  filter_depth=filter_depth_default):
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
    if filter_depth is not None:
        # mask (replace with NaNs) all values where depth > filter_depth_up_to:
        transect_ds = transect_ds.where(transect_ds.h < filter_depth)
    
    # condition 1: horizontal density gradient (away from coast) must be negative
    # this is determined per time over the entire transect
    drhodx = calculate_horizontal_density_gradient_along_transect(transect_ds)
    condition1 = np.nanmean(drhodx, axis=1) < 0. # taking mean over entire transect
    
    # condition 2: vertical density gradient > a minimum value (differs per ocean model)
    # AND the vertical density gradient must exceed this value for a minimum number of cells
    # the result is that DSWT is determined for the entire transect
    n_z_layers = len(transect_ds.z_rho)
    n_depth_layers = int(np.ceil(n_z_layers*drhodz_depth_p))
    l_drhodz = np.any(transect_ds.vertical_density_gradient[:, 0:n_depth_layers, :] > minimum_drhodz, axis=1) # [time, distance] (check for any along depth)
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
        condition2.append((n_consecutive_cells/n_used_cells) >= minimum_p_cells)
    condition2 = np.array(condition2)
    
    # DSWT along a transect when both condition 1 and condition 2 hold
    l_dswt = np.logical_and(condition1, condition2)
    
    return l_dswt, condition1, condition2, drhodz_max, drhodz_cells

def determine_dswt_along_multiple_transects(roms_ds:xr.Dataset, transects:dict,
                                            minimum_drhodz=minimum_drhodz_default,
                                            minimum_p_cells=minimum_p_cells_default,
                                            drhodz_depth_p=drhodz_depth_p_default,
                                            filter_depth=filter_depth_default) -> dict:
    
    transect_names = list(transects.keys())
    
    transects_dswt = {}
    for transect_name in transect_names:
        lon_land = transects[transect_name]['lon_land']
        lat_land = transects[transect_name]['lat_land']
        lon_ocean = transects[transect_name]['lon_ocean']
        lat_ocean = transects[transect_name]['lat_ocean']
        
        transect_ds = select_roms_transect(roms_ds, lon_land, lat_land, lon_ocean, lat_ocean)
        l_dswt, _, _, _, _ = determine_dswt_along_transect(transect_ds, minimum_drhodz=minimum_drhodz,
                                                 minimum_p_cells=minimum_p_cells, drhodz_depth_p=drhodz_depth_p,
                                                 filter_depth=filter_depth)
        
        transects_dswt[transect_name] = {'l_dswt': l_dswt,
                                         'lon_land': lon_land,
                                         'lat_land': lat_land,
                                         'lon_ocean': lon_ocean,
                                         'lat_ocean': lat_ocean}
        
    return transects_dswt    

def calculate_mean_dswt_along_all_transects(ds:xr.Dataset, transects:dict,
                                            minimum_drhodz=minimum_drhodz_default,
                                            minimum_p_cells=minimum_p_cells_default,
                                            drhodz_depth_p=drhodz_depth_p_default,
                                            filter_depth=filter_depth_default) -> float:
    transects_dswt = determine_dswt_along_multiple_transects(ds, transects,
                                                             minimum_drhodz=minimum_drhodz,
                                                             minimum_p_cells=minimum_p_cells,
                                                             drhodz_depth_p=drhodz_depth_p,
                                                             filter_depth=filter_depth)
    transect_names = list(transects_dswt.keys())
    
    l_dswt = np.zeros((len(transects_dswt[transect_names[0]]['l_dswt']), len(transect_names)))
    for i, t in enumerate(transect_names):
        l_dswt[:, i] = transects_dswt[t]['l_dswt']
    
    mean_dswt = np.nanmean(l_dswt)
    
    return mean_dswt

if __name__ == '__main__':
    roms_ds = load_roms_data('tests/data/cwa_20170222.nc', 'tests/data/cwa_grid.nc')
    l_dswt = determine_dswt_along_multiple_transects(roms_ds, 'tests/data/cwa_transects.json')