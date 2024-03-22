import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

import toml
from dataclasses import dataclass

@dataclass
class Config:
    drhodz_depth_percentage: float
    minimum_percentage_consecutive_cells: float
    filter_depth: float
    dswt_cross_shelf_transport_depth_range: float
    minimum_drhodz: float

def read_config(model:str, input_path='input/configs/main_config.toml') -> dict:
    full_config = toml.load(input_path)
    config = dict(full_config['DEFAULT'])
    for k,v in full_config[model].items():
        config[k] = v
        
    return Config(
        drhodz_depth_percentage = config['drhodz_depth_percentage'],
        minimum_percentage_consecutive_cells = config['minimum_percentage_consecutive_cells'],
        filter_depth = config['filter_depth'],
        dswt_cross_shelf_transport_depth_range = config['dswt_cross_shelf_transport_depth_range'],
        minimum_drhodz = config['minimum_drhodz']
    )
