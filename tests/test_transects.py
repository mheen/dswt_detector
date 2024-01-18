import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

import pytest
import xarray as xr
import numpy as np
import shapely
from generate_transects import convert_land_mask_to_polygons, get_largest_land_polygon, get_land_polygon
from generate_transects import get_continental_shelf_points, calculate_perpendicular_angles_to_shelf_points
from generate_transects import find_closest_land_point_at_angle

import matplotlib.pyplot as plt

mask = np.array([[1, 1, 1, 0],
                 [1, 1, 1, 0],
                 [0, 1, 0, 0],
                 [1, 1, 0, 0],
                 [1, 1, 1, 0]]) # 1: ocean, 0: land
lon = np.array([[114.0, 115.0, 116.0, 117.0],
                [114.0, 115.0, 116.0, 117.0],
                [114.0, 115.0, 116.0, 117.0],
                [114.0, 115.0, 116.0, 117.0],
                [114.0, 115.0, 116.0, 117.0]])
lat = np.array([[-33.0, -33.0, -33.0, -33.0],
                [-32.0, -32.0, -32.0, -32.0],
                [-31.0, -31.0, -31.0, -31.0],
                [-30.0, -30.0, -30.0, -30.0],
                [-29.0, -29.0, -29.0, -29.0]])
h = np.array([[400., 300., 200., 50.],
              [300., 200., 50., 0.],
              [300., 200., 50., 0.],
              [300., 200., 50., 0.],
              [200., 100., 50., 0.]])

def test_convert_land():
    land_polys = convert_land_mask_to_polygons(lon, lat, mask)
    x_island = np.array([114.0, 114.0, 115.0, 115.0, 114.0])
    y_island = np.array([-31.0, -30.0, -30.0, -31.0, -31.0])
    island_polygon = shapely.Polygon(list(zip(x_island, y_island)))
    assert land_polys[0] == island_polygon

def test_get_largest_polygon():
    land_polys = convert_land_mask_to_polygons(lon, lat, mask)
    largest_land = get_largest_land_polygon(land_polys)
    assert largest_land.area == 7.0
    
def test_get_land_polygon():
    land_poly = get_land_polygon(lon, lat, mask)
    assert land_poly.area == 7.0
    
def test_continental_shelf_points():
    shelf_lons, shelf_lats = get_continental_shelf_points(lon, lat, h)
    shelf_lons0 = np.array([116., 115.293, 115., 115., 114.586, 114.])
    shelf_lats0 = np.array([-33., -32.293, -31.414, -30.414, -29.586, -29.])
    assert (np.round(shelf_lons, 3) == shelf_lons0).all() & (np.round(shelf_lats, 3) == shelf_lats0).all()

def test_perpendicular_angles():
    shelf_lons, shelf_lats = get_continental_shelf_points(lon, lat, h)
    angles = calculate_perpendicular_angles_to_shelf_points(shelf_lons, shelf_lats)
    angles0 = np.array([0.865, 1.074, 1.436, 1.375, 1.013, 0.849])
    assert (np.round(angles, 3) == angles0).all()

def test_closest_land_point():
    land_poly = get_land_polygon(lon, lat, mask)
    shelf_lons, shelf_lats = get_continental_shelf_points(lon, lat, h)
    angles = calculate_perpendicular_angles_to_shelf_points(shelf_lons, shelf_lats)
    
    land_lons = []
    land_lats = []
    for i in range(len(shelf_lons)):
        land_lon, land_lat = find_closest_land_point_at_angle(shelf_lons[i], shelf_lats[i], angles[i], land_poly)
        land_lons.append(land_lon)
        land_lats.append(land_lat)
    land_lons = np.array(land_lons)
    land_lats = np.array(land_lats)
    land_lons0 = np.array([117., 117., 117., 116., 117.])
    land_lats0 = np.array([-32.148, -31.367, -31.144, -30.216, -28.081])
    assert (land_lons[~np.isnan(land_lons)] == land_lons0).all() & (np.round(land_lats[~np.isnan(land_lats)], 3) == land_lats0).all()
