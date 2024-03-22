from readers.read_ocean_data import select_roms_transect
from tools.roms import get_eta_xi_along_transect, find_eta_xi_covering_lon_lat_box
from tools.coordinates import get_bearing_between_points, get_distance_between_points
from tools import log

import numpy as np
import xarray as xr
from rasterio import features
import shapely
from pyproj import Geod
from warnings import warn
import json

import matplotlib.pyplot as plt
from plot_tools.basic_maps import plot_basic_map, plot_contours
import cartopy.crs as ccrs

def convert_land_mask_to_polygons(lon:np.ndarray[float], lat:np.ndarray[float], mask:np.ndarray[int]) -> list[shapely.Polygon]:
    land_mask = np.abs(mask-1).astype(np.int32) # assuming ocean points are 1 in mask!
    
    shapes = features.shapes(land_mask, mask=land_mask.astype(bool)) # mask so that only land polygons are returned
    
    # add extra column and row to lon and lat because polygon goes around edges
    # note: this is obviously not the best way to do this, but errors seem to be minor
    lon_extended = np.hstack((lon, np.expand_dims(lon[:, -1]+np.diff(lon, axis=1)[:, -1], 1)))
    lon_extended = np.vstack((lon_extended, lon_extended[-1, :]))
    lat_extended = np.hstack((lat, np.expand_dims(lat[:, -1], 1)))
    lat_extended = np.vstack((lat_extended, lat_extended[-1, :]+np.diff(lat_extended, axis=0)[-1, :]))
    
    land_polys = []
    for vec, _ in shapes:
        i, j = shapely.geometry.shape(vec).exterior.xy
        i = np.array(i).astype(int)
        j = np.array(j).astype(int)
        land_polys.append(shapely.Polygon(list(zip(lon_extended[j, i], lat_extended[j, i]))))
        
    return land_polys

def get_largest_land_polygon(land_polys:list[shapely.Polygon]) -> shapely.Polygon:
    geod = Geod(ellps='WGS84')
    
    areas = []
    for poly in land_polys:
        areas.append(abs(geod.geometry_area_perimeter(poly)[0]))
        
    i_max = np.where(np.array(areas) == max(areas))[0][0]
    
    return land_polys[i_max]

def get_land_polygon(lon:np.ndarray, lat:np.ndarray, mask:np.ndarray) -> shapely.Polygon:
    land_polys = convert_land_mask_to_polygons(lon, lat, mask)
    largest_land = get_largest_land_polygon(land_polys)
    
    return largest_land

def get_continental_shelf_points(lon:np.ndarray, lat:np.ndarray, h:np.ndarray,
                                 shelf_depth=200.) -> tuple[float, float]:
    
    # open and close dummy figure (unfortunately way to get this done using matplotlib)
    # !!! FIX !!! see if this can be done without matplotlib, maybe gdal?
    fig = plt.figure()
    ax = plt.axes()
    cs = ax.contour(lon, lat, h, levels=[shelf_depth])
    plt.close(fig)
    
    vertices = cs.collections[0].get_paths()[0].vertices
    shelf_line = shapely.LineString(vertices)
    
    # get evenly spaced points along line
    # aiming to get points spaced at around the size of the model grid cells
    # note: this is in degrees (shapely does not know coordinate system)
    dx = np.nanmean(np.unique(np.diff(lon, axis=1)))
    dy = np.nanmean(np.unique(np.diff(lat, axis=0)))
    ds = np.nanmin([dx, dy])
    point_distances = np.arange(0, shelf_line.length, ds)
    
    shelf_points = [shelf_line.interpolate(distance) for distance in point_distances] + [shelf_line.boundary.geoms[1]]
    
    shelf_lons = np.array([p.coords.xy[0][0] for p in shelf_points])
    shelf_lats = np.array([p.coords.xy[1][0] for p in shelf_points])
    
    return shelf_lons, shelf_lats

def calculate_perpendicular_angles_to_shelf_points(lon_ps:np.ndarray, lat_ps:np.ndarray) -> np.ndarray[float]:
    
    angle = [get_bearing_between_points(lon_ps[0], lat_ps[0], lon_ps[1], lat_ps[1])]
    for i in range(1, len(lon_ps)-1):
        angle.append(get_bearing_between_points(lon_ps[i-1], lat_ps[i-1], lon_ps[i+1], lat_ps[i+1]))
    angle.append(get_bearing_between_points(lon_ps[-2], lat_ps[-2], lon_ps[-1], lat_ps[-1]))
    
    angle = np.array(angle)
    
    perpendicular_angle = angle + 90. # !!! FIX !!! needs to be either +90 or -90 depending on direction
    
    return np.deg2rad(perpendicular_angle)

def find_closest_land_point_at_angle(lon_p:float, lat_p:float, angle:float,
                                     land_polygon:shapely.Polygon, dist=10000.) -> tuple[float]:
    '''Determines the closest point on land at a specific angle from an ocean point.
        
    Input
    lon_p: longitude coordinate of ocean point
    lat_p: latitude coordinate of ocean point
    angle: angle to find land at in radians
    land_polygon: polygon with land mass
    dist: length of line to intersect with land mass
    
    Returns
    lon_land: longitude coordinate of closest land point at requested angle
    lat_land: latitude coordinate of closest land point at requested angle
    
    Based on code by http://geoexamples.blogspot.com/2014/08/shortest-distance-to-geometry-in.html
    '''
    
    line = shapely.LineString([(lon_p, lat_p), (lon_p + dist * np.sin(angle), lat_p + dist * np.cos(angle))])
    
    difference = line.difference(land_polygon) # splits line into two with polygon removing lines

    if difference.geom_type == 'MultiLineString': # first line is from point to land polygon
        lon_land = difference.geoms[0].coords.xy[0][-1]
        lat_land = difference.geoms[0].coords.xy[1][-1]
    elif difference.geom_type == 'LineString': # if line is short, there is only one line
        lon_land = difference.coords[-1][0]
        lat_land = difference.coords[-1][1]
    else:
        raise ValueError(f'Difference did not return a MultiLineString or a LineString as expected: {difference.geom_type}')
        
    if lon_land == line.coords[-1][0] or lat_land == line.coords[-1][1]:
        # 'land' point is just end point of line: did not cross land polygon
        warn(f'''No land point found for {(lon_p, lat_p, angle)}, returning NaN values.
             If this happens for all points, consider increasing dist={dist}.''')
        lon_land = np.nan
        lat_land = np.nan
    
    return lon_land, lat_land

def generate_transects_json_file(ds:xr.Dataset, output_path:str):
    
    land_polygon = get_land_polygon(ds.lon_rho.values, ds.lat_rho.values, ds.mask_rho.values)
    shelf_lons, shelf_lats = get_continental_shelf_points(ds.lon_rho.values, ds.lat_rho.values, ds.h.values)
    angles = calculate_perpendicular_angles_to_shelf_points(shelf_lons, shelf_lats)
    
    transects = {}
    for i in range(len(shelf_lons)):
        lon_land, lat_land = find_closest_land_point_at_angle(shelf_lons[i], shelf_lats[i], angles[i], land_polygon)
        if np.isnan(lon_land) or np.isnan(lat_land):
            log.info(f'No land point found for shelf point: {shelf_lons[i], shelf_lats[i]}')
            continue
        transects[f't{i}'] = {'lon_land': lon_land, 'lat_land': lat_land, 'lon_ocean': shelf_lons[i], 'lat_ocean': shelf_lats[i]}
    
    log.info(f'Writing transects to json file: {output_path}')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transects, f, ensure_ascii=False, indent=4)

# !!! FIX !!! Determining the transect "width" is a bit of a hack,
# see if it works out alright (compared to overall cross-shelf transport
# just along a contour line) and if not see if there is a better
# way to do this.
def calculate_transect_width(transects:dict) -> dict:
    '''Approximating transect width by determining the distance
    between consecutive transect ocean starting points. I am using
    ocean points because the transects are built from the continental
    shelf towards land and should be pretty much equidistant along
    the shelf but not necessarily along land, depending on the angle
    between the shelf and land. This should be about the same size
    as a grid cell because that is how the transects are built.'''
    
    transect_names = list(transects.keys())
    transect_names.sort()
    
    for i in range(len(transect_names)-1):
        lon1 = transects[transect_names[i]]['lon_ocean']
        lat1 = transects[transect_names[i]]['lat_ocean']
        lon2 = transects[transect_names[i+1]]['lon_ocean']
        lat2 = transects[transect_names[i+1]]['lat_ocean']
        
        if i == 0:
            ds = get_distance_between_points(lon1, lat1, lon2, lat2)
        else:
            lon0 = transects[transect_names[i-1]]['lon_ocean']
            lat0 = transects[transect_names[i-1]]['lat_ocean']
            ds0 = get_distance_between_points(lon0, lat0, lon1, lat1)
            ds1 = get_distance_between_points(lon1, lat1, lon2, lat2)
            ds = (ds0+ds1)/2
        
        transects[transect_names[i]]['width'] = ds
    
    transects[transect_names[i+1]]['width'] = ds # same width for last as second-to-last transect
        
    return transects

def get_transects_dict_from_json(transects_file:str) -> dict:

    with open(transects_file, 'r') as f:
        all_transects = json.load(f)
    transect_names = list(all_transects.keys())
    
    transects = {}
    for i in range(len(transect_names)):
        lon_land = all_transects[transect_names[i]]['lon_land']
        lat_land = all_transects[transect_names[i]]['lat_land']
        lon_ocean = all_transects[transect_names[i]]['lon_ocean']
        lat_ocean = all_transects[transect_names[i]]['lat_ocean']
       
        transects[transect_names[i]] = {'lon_land': lon_land, 'lat_land': lat_land,
                                        'lon_ocean': lon_ocean, 'lat_ocean': lat_ocean}
    
    transects = calculate_transect_width(transects) # add width to transects dict
        
    return transects

def get_transects_in_lon_lat_range(transects_file:str,
                                   lon_range:list[float],
                                   lat_range:list[float]) -> dict:
    
    all_transects = get_transects_dict_from_json(transects_file)
    
    if lon_range is None and lat_range is None:
        return all_transects

    transects = {}
    for t in list(all_transects.keys()):
        l_lon_land = np.logical_and(lon_range[0] <= all_transects[t]['lon_land'], all_transects[t]['lon_land'] <= lon_range[1])
        l_lon_ocean = np.logical_and(lon_range[0] <= all_transects[t]['lon_ocean'], all_transects[t]['lon_ocean'] <= lon_range[1])
        l_lon = np.logical_and(l_lon_land, l_lon_ocean)
        
        l_lat_land = np.logical_and(lat_range[0] <= all_transects[t]['lat_land'], all_transects[t]['lat_land'] <= lat_range[1])
        l_lat_ocean = np.logical_and(lat_range[0] <= all_transects[t]['lat_ocean'], all_transects[t]['lat_ocean'] <= lat_range[1])
        l_lat = np.logical_and(l_lat_land, l_lat_ocean)
        
        l_range = np.logical_and(l_lon, l_lat)
        
        if l_range == True:
            transects[t] = all_transects[t]
        else:
            continue
    
    return transects

def get_specific_transect_data(roms_ds:xr.Dataset, transects:dict, transect_name:str) -> xr.Dataset:
    transect = next(transects[item] for item in transects if item==transect_name)
    
    transect_ds = select_roms_transect(roms_ds, transect['lon_land'], transect['lat_land'],
                                       transect['lon_ocean'], transect['lat_ocean'])
    return transect_ds
    
# add plotting function to check transects?

# add function that determines how many grid cells are not covered by transects?