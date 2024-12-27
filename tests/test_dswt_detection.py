import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

import pytest
from tools.config import read_config
from transects import read_transects_dict_from_json
from readers.read_ocean_data import load_roms_data, select_roms_transect_from_known_coordinates
from dswt.dswt_detection import determine_dswt_along_transect

transects_file = f'input/transects/test_transects.json'
transects = read_transects_dict_from_json(transects_file)

config = read_config('test')

ds_roms = load_roms_data('tests/data/2017/test_20170607_03__his.nc', grid_file='tests/data/grid.nc')

transect_name = 't162'

eta = transects[transect_name]['eta']
xi = transects[transect_name]['xi']

transect_ds = select_roms_transect_from_known_coordinates(ds_roms, eta, xi)

def test_detection():
    l_dswt, _, _, _, _ = determine_dswt_along_transect(transect_ds, config)
    l_dswt0 = [True, True, True, True, True, True, True, True]
    assert (l_dswt == l_dswt0).all()
