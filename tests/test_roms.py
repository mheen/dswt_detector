import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

import pytest
import xarray as xr
import numpy as np
from tools.roms import get_eta_xi_of_lon_lat_point, find_eta_xi_covering_lon_lat_box, get_eta_xi_along_transect
from tools.roms import get_z
from readers.read_ocean_data import load_roms_data

grid_file = 'tests/data/grid.nc'
ds_grid = xr.load_dataset(grid_file)
lon_grid = ds_grid.lon_rho.values
lat_grid = ds_grid.lat_rho.values

ds_roms = load_roms_data('tests/data/2017/test_20170607_03__his.nc', grid_file=grid_file)

def test_point():
    eta, xi = get_eta_xi_of_lon_lat_point(lon_grid, lat_grid, 115.70, -31.76)
    assert (eta, xi) == (151, 73)
    
def test_box():
    xi0, xi1, eta0, eta1 = find_eta_xi_covering_lon_lat_box(lon_grid, lat_grid, [114.0, 116.0], [-32.0, -30.0])
    assert (xi0, xi1, eta0, eta1) == (42, 152, 62, 124)
        
def test_z_rho():
    z_rho = get_z(ds_grid.Vtransform.values, ds_grid.s_rho.values, ds_grid.h.values, ds_grid.Cs_r.values, ds_grid.hc.values)
    assert np.round(z_rho[0, 0, 0], 2) == -2960.9

def test_u_east_conversion():
    assert np.round(ds_roms.u_eastward[0, 0, 0, 0].values, 3) == -0.031
    
def test_v_north_conversion():
    assert np.round(ds_roms.v_northward[0, 0, 0, 0].values, 3) == 0.132