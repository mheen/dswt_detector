from geopy.distance import geodesic
import numpy as np

def get_distance_between_arrays(lon1:np.ndarray, lat1: np.ndarray,
                                lon2:np.ndarray, lat2:np.ndarray) -> np.ndarray:
    # check that array sizes are all the same
    if lon1.shape!=lat1.shape or lon1.shape!=lon2.shape or lon1.shape!=lat2.shape:
        raise ValueError('Dimensions of longitudes and latitudes for points are not the same.')

    lon1 = lon1.flatten()
    lat1 = lat1.flatten()
    lon2 = lon2.flatten()
    lat2 = lat2.flatten()

    distance = []

    for i in range(len(lon1)):
        distance.append(get_distance_between_points(lon1[i], lat1[i], lon2[i], lat2[i]))
    
    return np.array(distance)

def get_distance_between_points(lon1:float, lat1:float, lon2:float, lat2:float) -> float:
    pos1 = (lat1, lon1)
    pos2 = (lat2, lon2)
    distance = geodesic(pos2, pos1).meters
    return distance

def get_closest_index(full_array:np.ndarray, target:np.ndarray) -> np.ndarray:
    '''Returns the index of the closest element in an array.
    The requested target can be a float, list, or np.ndarray.
    The closest index will be an int or list of ints.'''
    i_sort = np.argsort(full_array)
    sorted_array = full_array[i_sort]
    idx = sorted_array.searchsorted(target)
    idx = np.clip(idx, 1, len(sorted_array)-1)
    left = sorted_array[idx-1]
    right = sorted_array[idx]
    idx -= target-left < right-target
    i_unsorted = i_sort[idx]
    return i_unsorted