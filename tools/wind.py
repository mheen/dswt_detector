import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.arrays import get_closest_index
import numpy as np

def convert_u_v_to_meteo_vel_dir(u:np.ndarray, v:np.ndarray) -> tuple:
    vel = np.sqrt(u**2+v**2)

    dir = np.mod(180+180/np.pi*np.arctan2(u, v), 360)

    return vel, dir

def get_wind_dir_and_text() -> tuple:
    dir = [0, 45, 90, 135, 180, 225, 270, 315]
    text = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    return dir, text

def get_lon_lat_range_indices(lon:np.ndarray, lat:np.ndarray,
                              lon_range:list, lat_range:list) -> tuple[int, int, int, int]:
    i0 = get_closest_index(lon[0, :], lon_range[0])
    i1 = get_closest_index(lon[0, :], lon_range[1])
    j0 = get_closest_index(lat[:, 0], lat_range[0])
    j1 = get_closest_index(lat[:, 0], lat_range[1])
    return i0, i1, j0, j1