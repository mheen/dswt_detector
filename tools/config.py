import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

import toml
from dataclasses import dataclass

@dataclass
class Config:
    transect_contours: list
    angle_tolerance: float
    max_distance_contours: float
    drhodz_depth_percentage: float
    minimum_percentage_consecutive_cells: float
    filter_depth: float
    dswt_cross_shelf_transport_depth_range: float
    minimum_drhodz: float
    cross_shelf_transport_depth: float
    cross_shelf_bottom_layers_percentage: float
    cross_shelf_surface_layers_percentage: float
    cross_shelf_interior_layers_percentage: float

def print_config(config:Config):
    print(f'''Current configuration settings are:
          drho/dz threshold: {config.minimum_drhodz} kg/m$^3$/m,
          for a minimum of {config.minimum_percentage_consecutive_cells*100}% consecutive grid cells,
          over the {config.drhodz_depth_percentage*100}% of the bottom grid cells.''')

def read_config(model:str, input_path='input/configs/main_config.toml') -> dict:
    full_config = toml.load(input_path)
    config = dict(full_config['DEFAULT'])
    for k,v in full_config[model].items():
        config[k] = v
        
    dswt_cross_shelf_transport_depth_range = config['dswt_cross_shelf_transport_depth_range']
    if dswt_cross_shelf_transport_depth_range[0] > dswt_cross_shelf_transport_depth_range[1]:
        # swap order so that lower depth value comes first
        dswt_cross_shelf_transport_depth_range = [dswt_cross_shelf_transport_depth_range[1],
                                                  dswt_cross_shelf_transport_depth_range[0]]
        
    if config['cross_shelf_interior_layers_percentage'] is None:
        config['cross_shelf_interior_layers_percentage'] = 1.0-config['cross_shelf_bottom_layers_percentage']-config['cross_shelf_surface_layers_percentage']
    else: # check if sum of layers is 1, otherwise adjust interior layers to match
        if config['cross_shelf_bottom_layers_percentage']+config['cross_shelf_surface_layers_percentage']+config['cross_shelf_interior_layers_percentage'] != 1.0:
            config['cross_shelf_interior_layers_percentage'] = 1.0-config['cross_shelf_bottom_layers_percentage']-config['cross_shelf_surface_layers_percentage']
 
    return Config(
        transect_contours = config['transect_contours'],
        angle_tolerance = config['angle_tolerance'],
        max_distance_contours = config['max_distance_contours'],
        drhodz_depth_percentage = config['drhodz_depth_percentage'],
        minimum_percentage_consecutive_cells = config['minimum_percentage_consecutive_cells'],
        filter_depth = config['filter_depth'],
        dswt_cross_shelf_transport_depth_range = dswt_cross_shelf_transport_depth_range,
        minimum_drhodz = config['minimum_drhodz'],
        cross_shelf_transport_depth = config['cross_shelf_transport_depth'],
        cross_shelf_bottom_layers_percentage = config['cross_shelf_bottom_layers_percentage'],
        cross_shelf_surface_layers_percentage = config['cross_shelf_surface_layers_percentage'],
        cross_shelf_interior_layers_percentage = config['cross_shelf_interior_layers_percentage']
    )
