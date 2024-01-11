from ocean_model_data import load_roms_data, select_roms_subset, select_roms_transect
import numpy as np
import xarray as xr

roms_ds = load_roms_data('tests/data/', grid_file='tests/data/cwa_grid.nc', files_contain='cwa_test_dswt')
transect_ds = select_roms_transect(roms_ds, 115.7, -31.76, 115.26, -31.95)
transect_ds.density[0, :, :].plot(x='distance', y='z_rho')

def calculate_depth_mean_density(roms_ds:xr.Dataset) -> np.ndarray:
    
    delta_z = np.diff(roms_ds.z_w.vaues, axis=0)
    depth_mean_density = np.sum(roms_ds.density.values*delta_z, axis=1)/roms_ds.h.values
    
    return depth_mean_density

def calculate_vertical_density_gradient(roms_ds:xr.Dataset) -> np.ndarray:
    delta_z = np.diff(roms_ds.z_rho.values, axis=0)
    drhodz = np.diff(roms_ds.density, axis=1)/delta_z
    
    return drhodz

