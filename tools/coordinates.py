from geographiclib.geodesic import Geodesic
import numpy as np
import json

def get_angle_between_bearings(bearing1:float, bearing2:float) -> float:
    # bearings in degrees
    angle = abs(bearing1 - bearing2) % 360 # % is modulo operator: gives remainder
    if angle > 180.0:
        angle = 360. - angle
    return angle # angle in degrees

def get_point_from_bearing_and_distance(lon0:float, lat0:float, bearing:float, dist:float) -> float:
    geod = Geodesic.WGS84
    g = geod.Direct(lat0, lon0, bearing, dist)
    return g['lon2'], g['lat2']

def get_bearing_between_points(lon1, lat1, lon2, lat2):
    geod = Geodesic.WGS84
    g = geod.Inverse(lat1, lon1, lat2, lon2)
    return g['azi1'] # bearing in degrees

def get_distance_between_points(lon1:float, lat1:float, lon2:float, lat2:float) -> float:
    geod = Geodesic.WGS84
    g = geod.Inverse(lat1, lon1, lat2, lon2)
    return g['s12'] # distance in meters

def get_points_on_line_between_points(lon1:float, lat1:float, lon2:float, lat2:float, ds:float) -> tuple:
    geod = Geodesic.WGS84
    l = geod.InverseLine(lat1, lon1, lat2, lon2)
    
    n = int(np.ceil(l.s13/ds))
    
    lons = []
    lats = []
    for i in range(n+1):
        s = min(ds*i, l.s13)
        g = l.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL) # Geodesic.LONG_UNROLL stops longitude jumping when crossing dateline
        lons.append(g['lon2'])
        lats.append(g['lat2'])
    
    return np.array(lons), np.array(lats)

def get_index_closest_point(lon, lat, lon0, lat0, n_closest=1):
    distance = []
    for i in range(len(lon)):
        distance.append(get_distance_between_points(lon[i], lat[i], lon0, lat0))
    distance = np.array(distance)
    i_closest = np.where(distance == np.nanmin(distance))[0][0]
    i_remove = i_closest
    i_closest = [i_closest]
    while n_closest > 1:
        distance = np.delete(distance, i_remove)
        i_remove = np.where(distance == np.nanmin(distance))[0][0]
        i_closest.append(i_remove)        
        n_closest -= 1
    return i_closest

def convert_lon_360_to_180(lon):
    lon[lon>180] = lon[lon>180]-360
    return lon