import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

import pytest
import xarray as xr
import numpy as np
from datetime import datetime
from transects import read_transects_dict_from_json
from readers.read_ocean_data import load_roms_data, select_roms_transect_from_known_coordinates
from tools.config import read_config
from dswt.cross_shelf_transport import calculate_down_transect_velocity_component, calculate_dswt_cross_shelf_transport_along_transect

transects_file = f'input/transects/test_transects.json'
transects = read_transects_dict_from_json(transects_file)

config = read_config('test')

ds_roms = load_roms_data('tests/data/2017/test_20170607_03__his.nc', grid_file='tests/data/grid.nc')

transect_name = 't162'

eta = transects[transect_name]['eta']
xi = transects[transect_name]['xi']

transect_ds = select_roms_transect_from_known_coordinates(ds_roms, eta, xi)

def test_down_transect_velocity():
    down_vel = calculate_down_transect_velocity_component(0.1, 0.0, 114.0, -32.0, 116.0, -32.0)
    assert np.round(down_vel, 1) == 0.1

def test_cross_shelf_transport():
    l_dswt = np.array([True])
    transport, _, _, _, _, _, _ = calculate_dswt_cross_shelf_transport_along_transect(transect_ds, l_dswt, config)
    assert np.round(np.nanmean(transport), 0) == 4572.0
