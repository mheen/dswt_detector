import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

import pytest
import xarray as xr
import numpy as np
from tools.roms import get_eta_xi_of_lon_lat_point, find_eta_xi_covering_lon_lat_box, get_eta_xi_along_transect
from tools.roms import get_z
from ocean_model_data import select_input_files, load_roms_data

cwa_grid = xr.load_dataset('tests/data/cwa_grid.nc')
cwa_lon = cwa_grid.lon_rho.values
cwa_lat = cwa_grid.lat_rho.values

cwa_ds = load_roms_data('tests/data/', files_contain='cwa_test_dswt', grid_file='tests/data/cwa_grid.nc')

class TestCoords:
    def test_point(self):
        eta, xi = get_eta_xi_of_lon_lat_point(cwa_lon, cwa_lat, 115.70, -31.76)
        assert (eta, xi) == (477, 111)
        
    def test_box(self):
        xi0, xi1, eta0, eta1 = find_eta_xi_covering_lon_lat_box(cwa_lon, cwa_lat, [114.0, 116.0], [-32.0, -30.0])
        assert (xi0, xi1, eta0, eta1) == (368, 479, 100, 208)
        
    def test_transect(self):
        etas, xis = get_eta_xi_along_transect(cwa_lon, cwa_lat, 115.7, -31.76, 115.26, -31.95, 500.)
        etas0 = np.array([111, 111, 111, 111, 111, 110, 110, 110, 110, 110, 110, 109, 109,
                          109, 109, 109, 109, 108, 108, 108, 108, 108, 108, 107, 107, 107,
                          107, 107, 107, 106, 106, 106, 106, 106, 106, 105, 105])
        xis0 = np.array([477, 476, 475, 474, 473, 473, 472, 471, 470, 469, 468, 468, 467,
                         466, 465, 464, 463, 463, 462, 461, 460, 459, 458, 458, 457, 456,
                         455, 454, 453, 453, 452, 451, 450, 449, 448, 448, 447])
        assert (etas == etas0).all() & (xis == xis0).all()
        
def test_z_rho():
    z_rho = get_z(cwa_grid.Vtransform.values, cwa_grid.s_rho.values, cwa_grid.h.values, cwa_grid.Cs_r.values, cwa_grid.hc.values)
    assert np.round(z_rho[0, 0, 0], 2) == -4901.25 and np.round(z_rho[-1, 320, 240], 2) == -3.97
    
class TestLoadRoms:    
    def test_input_file_selection(self):
        paths = select_input_files('tests/data/', file_contains='cwa_test_dswt')
        assert paths == ['tests/data/cwa_test_dswt.nc']

    def test_grid_file_error(self):
        with pytest.raises(ValueError):
            load_roms_data('tests/data/', files_contain='cwa_test_dswt')
            
def test_u_east_conversion():
    assert np.round(cwa_ds.u_east[0, 0, 0, 0].values, 3) == -0.065 and np.round(cwa_ds.u_east[4, 10, 320, 240].values, 3) == -0.045
    
def test_v_north_conversion():
    assert np.round(cwa_ds.v_north[0, 0, 0, 0].values, 3) == -0.025 and np.round(cwa_ds.v_north[4, 10, 320, 240].values, 3) == 0.062