from ocean_model_data import load_roms_data, select_roms_subset, select_roms_transect
from tools.coordinates import get_distance_between_points
from plot_tools.plot_cycler import plot_cycler
from itertools import groupby
import numpy as np
import xarray as xr
import json

import matplotlib.pyplot as plt

def calculate_horizontal_density_gradient_along_transect(transect_ds:xr.Dataset) -> float:
    drho = np.diff(transect_ds.depth_mean_density.values, axis=1)
    dx = np.diff(transect_ds.distance.values)
    
    drhodx = drho/dx
    
    # add dummy value at end for size consistency
    # !!! FIX !!! find a better way to do this
    drhodx = np.hstack((drhodx, np.expand_dims(drhodx[:, -1], 1)))

    return drhodx

def determine_dswt_along_transect(transect_ds:xr.Dataset,
                                  minimum_drhodz=0.02,
                                  minimum_p_cells=0.3,
                                  filter_depth=100.):
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
       of the total number of cells along the transect: minimum_p_cells). This also
       returns one bool per time over the entire transect.
       The default value for minimum_p_cells = 0.3 (30%), which should hopefully work
       independent of the ocean model used.
    
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
    l_drhodz = np.any(transect_ds.vertical_density_gradient > minimum_drhodz, axis=1) # [time, distance] (check for any along depth)
    n_used_cells = sum(~np.isnan(transect_ds.h))
    condition2 = []
    for t in range(len(transect_ds.ocean_time)):
        consecutive_true_lengths = [len(list(v)) for k, v in groupby(l_drhodz.values[t, :]) if k == True]
        if len(consecutive_true_lengths) != 0:
            n_consecutive_cells = max(consecutive_true_lengths)
        else:
            n_consecutive_cells = 0
        condition2.append((n_consecutive_cells/n_used_cells) >= minimum_p_cells)
    condition2 = np.array(condition2)
    
    # DSWT along a transect when both condition 1 and condition 2 hold
    l_dswt = np.logical_and(condition1, condition2)
    
    return l_dswt

def determine_dswt_along_multiple_transects(roms_ds:xr.Dataset, transects_file:str,
                                            minimum_drhodz=0.02,
                                            minimum_p_cells=0.30,
                                            filter_depth=100.) -> np.ndarray[bool]:
    
    with open(transects_file, 'r') as f:
        all_transects = json.load(f)
    transect_names = list(all_transects.keys())
    
    l_dswt = []
    for t in transect_names:
        lon_land = all_transects[t]['lon_land']
        lat_land = all_transects[t]['lat_land']
        lon_ocean = all_transects[t]['lon_ocean']
        lat_ocean = all_transects[t]['lat_ocean']
        
        transect_ds = select_roms_transect(roms_ds, lon_land, lat_land, lon_ocean, lat_ocean)
        l_dswt_t = determine_dswt_along_transect(transect_ds, minimum_drhodz=minimum_drhodz,
                                                 minimum_p_cells=minimum_p_cells,
                                                 filter_depth=filter_depth)
        
        l_dswt.append(l_dswt_t)
        
    return l_dswt

def plot_cycling_transect(transect_ds:xr.Dataset, l_dswt=None, t_interval=1,
                          parameter='temp', vmin=20., vmax=22., cmap='RdYlBu_r',
                          filter_depth=100.) -> plt.axes:
    '''Plots a pcolormap of a transect which can be cycled through in time using arrow keys.'''

    if parameter not in transect_ds.variables:
        raise ValueError(f'Unknown parameter requested: {parameter}')

    def single_plot(fig, req_time):
        
        plot_ds = transect_ds.sel(ocean_time=req_time, method='nearest')
        t = list(transect_ds.ocean_time.values).index(transect_ds.sel(ocean_time=req_time, method='nearest').ocean_time.values)
        
        if len(plot_ds[parameter].shape) < 2:
            raise ValueError(f'Requested parameter does not have a depth component.')
        
        if l_dswt is not None:
            dswt = f', DSWT: {l_dswt[t]}'
        else:
            dswt = ''
        title = f'{plot_ds.ocean_time.values}{dswt}'
        
        ax = plt.axes()
        plot_ds[parameter].plot(x='distance', y='z_rho', vmin=vmin, vmax=vmax, cmap=cmap)
        ax.fill_between(plot_ds.distance.values, -plot_ds.h.values, np.nanmin(plot_ds.z_rho.values), edgecolor='k', facecolor='#989898')
        if filter_depth is not None:
            ax.fill_between(plot_ds.distance.values, -filter_depth, np.nanmin(plot_ds.z_rho.values), facecolor='#ffffff', alpha=0.5)
        
        ax.set_xlim([0, np.nanmax(plot_ds.distance.values)])
        ax.set_ylim([np.nanmin(plot_ds.z_rho.values), 0])

        ax.set_title(title)
    
    t = np.arange(0, len(transect_ds.ocean_time), t_interval)
    time = transect_ds.ocean_time.values[t]

    fig = plot_cycler(single_plot, time)
    plt.show()

def plot_cycling_vertical_density_gradients(transect_ds:xr.Dataset, t_interval=1,
                                            filter_depth=100.) -> plt.axes:
    '''Plots vertical profiles of drho/dz for all points along a transect.
    Can be cycled through in time using arrow keys. Use this to check your required minimum drho/dz
    values to determine if there is DSWT occurring or not.'''

    def single_plot(fig, req_time):
        
        plot_ds = transect_ds.sel(ocean_time=req_time, method='nearest')
        if filter_depth is not None:
            l_h = plot_ds.h.values <= filter_depth
        else:
            l_h = np.ones(plot_ds.distance.shape).astype(bool)
        
        title = f'{plot_ds.ocean_time.values}'
        
        ax = plt.axes()
        for i in range(len(plot_ds.distance)):
            if l_h[i] == True:
                linestyle = '-'
            else:
                linestyle = ':'
            plot_ds.vertical_density_gradient[:, i].plot(y='z_rho', linestyle=linestyle)

        ax.set_title(title)
    
    t = np.arange(0, len(transect_ds.ocean_time), t_interval)
    time = transect_ds.ocean_time.values[t]

    fig = plot_cycler(single_plot, time)
    plt.show()

if __name__ == '__main__':
    roms_ds = load_roms_data('tests/data/', grid_file='tests/data/cwa_grid.nc', files_contain='cwa_20170613')
    transect_ds = select_roms_transect(roms_ds, 115.7, -31.76, 115.26, -31.95)
    l_dswt = determine_dswt_along_transect(transect_ds)
    
    plot_cycling_vertical_density_gradients(transect_ds)
    
    plot_cycling_transect(transect_ds, l_dswt)
    
    