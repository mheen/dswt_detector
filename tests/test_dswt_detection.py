import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

import pytest
from ocean_model_data import load_roms_data, select_roms_transect
from dswt_detection import determine_dswt_along_transect

transect_coords1 = [115.7, -31.76, 115.26, -31.95]

def test_cwa_roms_20170613():
    roms_ds = load_roms_data('tests/data/', grid_file='tests/data/cwa_grid.nc', files_contain='cwa_20170613')
    transect_ds = select_roms_transect(roms_ds, transect_coords1[0], transect_coords1[1],
                                       transect_coords1[2], transect_coords1[3])
    l_dswt = determine_dswt_along_transect(transect_ds, minimum_drhodz=0.02, minimum_p_cells=0.3, filter_depth=100.)
    l_dswt0 = [True, True, True, True, True, True, True, True]
    assert (l_dswt == l_dswt0).all()
    
def test_cwa_roms_20170623():
    roms_ds = load_roms_data('tests/data/', grid_file='tests/data/cwa_grid.nc', files_contain='cwa_20170623')
    transect_ds = select_roms_transect(roms_ds, transect_coords1[0], transect_coords1[1],
                                       transect_coords1[2], transect_coords1[3])
    l_dswt = determine_dswt_along_transect(transect_ds, minimum_drhodz=0.02, minimum_p_cells=0.3, filter_depth=100.)
    l_dswt0 = [False, False, False, False, False, False, False, False]
    assert (l_dswt == l_dswt0).all()
    
def test_ozroms_20170613():
    roms_ds = load_roms_data('tests/data/', files_contain='ozroms_20170613')
    transect_ds = select_roms_transect(roms_ds, transect_coords1[0], transect_coords1[1],
                                       transect_coords1[2], transect_coords1[3])
    l_dswt = determine_dswt_along_transect(transect_ds, minimum_drhodz=0.01, minimum_p_cells=0.3, filter_depth=100.)
    l_dswt0 = [True]
    assert (l_dswt == l_dswt0).all()
    
def test_ozroms_20170623():
    roms_ds = load_roms_data('tests/data/', files_contain='ozroms_20170623')
    transect_ds = select_roms_transect(roms_ds, transect_coords1[0], transect_coords1[1],
                                       transect_coords1[2], transect_coords1[3])
    l_dswt = determine_dswt_along_transect(transect_ds, minimum_drhodz=0.01, minimum_p_cells=0.3, filter_depth=100.)
    l_dswt0 = [False]
    assert (l_dswt == l_dswt0).all()