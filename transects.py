from tools.config import Config, read_config
from tools.roms import get_eta_xi_of_lon_lat_point
from tools.coordinates import get_bearing_between_points, get_point_from_bearing_and_distance, get_angle_between_bearings, get_index_closest_point
from tools import log
from tools.files import get_dir_from_json

import numpy as np
import xarray as xr
import shapely
from warnings import warn
import json
import matplotlib.pyplot as plt

def get_depth_contours(lon:np.ndarray, lat:np.ndarray, h:np.ndarray,
                       depth_contours:list[float]) -> list[shapely.LineString]:
    '''Function determines depth contours at depths specified in config
    and returns them as a list of shapely.LineStrings. These
    are used to fit perpendicular transects as best as possible
    to different depths.
    
    This function currently uses matplotlib to determine contours.
    This is a bit of a hacky solution, since it requires creating
    a dummy figure and then closing it again. However, it also
    seems the most straightforward way in which to do this.
    It is worth considering changing this at some point though.
    
    Input:
    lon: numpy 2d array with longitude points
    lat: numpy 2d array with latitude points
    h: numpy 2d array with depths (bathymetry)
    depth_contours: list of contour depths to find
        
    Output:
    list of shapely.LineString with contours'''
        
    if depth_contours[0] < np.ceil(np.nanmin(h)):
        depth_contours[0] = np.ceil(np.nanmin(h))
   
    def _get_linestring_of_longest_line(path):
        codes = path.codes
        all_vertices = path.vertices
        
        # new path starts at code 1
        # only use vertices up to second code 1
        i_start = np.where(codes == 1)[0]
        if len(i_start) == 1:
            vertices = all_vertices
        else:
            vertices = all_vertices[:i_start[1]]
            
        contour = shapely.LineString(vertices)
        return contour
        
    fig = plt.figure() # dummy figure that is closed later
    ax = plt.axes()
    cs = ax.contour(lon, lat, h, levels=depth_contours)
    plt.close()
    
    contours = [_get_linestring_of_longest_line(path) for path in cs.get_paths()]
    
    return contours

def get_starting_points(contour:shapely.LineString, ds:float) -> tuple[np.ndarray[float], np.ndarray[float]]:
    '''Gets starting points along specified contour (should be the shallowest)
    that are evenly spaced along this contour at a distance ds (should be the grid cell size).
    Returns coordinates of these points.'''
    
    point_distances = np.arange(0, contour.length, ds)
    points = [contour.interpolate(distance) for distance in point_distances] + [contour.boundary.geoms[1]]
    
    lon_ps = np.array([p.coords.xy[0][0] for p in points])
    lat_ps = np.array([p.coords.xy[1][0] for p in points])
    
    return lon_ps, lat_ps

def get_perpendicular_angle(lon0:float, lat0:float, lon1:float, lat1:float) -> float:
    '''Returns angle of perpendicular line to two points in degrees.'''
    angle = get_bearing_between_points(lon0, lat0, lon1, lat1)
        
    if 0.0 <= angle <= 90.0:
        perp_angle = angle - 90.0
    elif -90.0 < angle < 0.0:
        perp_angle = angle + 90.0
    elif 90.0 < angle <= 180.0:
        perp_angle = angle - 90.0
    elif -180.0 <= angle <= -90.0:
        perp_angle = angle + 90.0
    else:
        raise ValueError(f'No perpendicular angle for {angle}')
        
    return perp_angle

def get_equivalent_angle(angle:float) -> float:
    '''Returns the "equivalent" angle in degrees.
    This is the angle of a line oriented 180 degrees
    in the opposite direction.'''
    
    if 0.0 <= angle <= 180.0:
        eq_angle = angle - 180.0
    elif -180.0 <= angle < 0.0:
        eq_angle = angle + 180.0
    else:
        raise ValueError(f'No equivalent angle for {angle}')
    
    return eq_angle

def smooth_angle(angle:float, angle_previous:float, diff_tolerance:float) -> float:
    
    eq_angle_previous = get_equivalent_angle(angle_previous)
    
    diff1 = get_angle_between_bearings(angle, angle_previous)
    diff2 = get_angle_between_bearings(angle, eq_angle_previous)
    
    if diff1 < diff2:
        if diff1 > diff_tolerance:
            smooth_angle = np.nanmean([angle, angle_previous])
        else:
            smooth_angle = angle
    else:
        if diff2 > diff_tolerance:
            smooth_angle = np.nanmean([angle, eq_angle_previous])
        else:
            smooth_angle = angle
            
    return smooth_angle

def create_line_at_angle_to_point(lon:float, lat:float,
                                  perp_angle:float, # in degrees
                                  config:Config
                                  ):
    
    point0 = get_point_from_bearing_and_distance(lon, lat, perp_angle, config.max_distance_contours)
    point1 = get_point_from_bearing_and_distance(lon, lat, perp_angle, -config.max_distance_contours)
    line = shapely.LineString([point0, point1])
    
    return line

def get_intersection_point_between_lines(line:shapely.LineString,
                                         target_line:shapely.LineString,
                                         lon_p:float, lat_p:float) -> tuple[float]:
        
    difference = line.difference(target_line) # splits line into segments

    if difference.geom_type == 'MultiLineString':
        x = np.array([g.coords.xy[0] for g in difference.geoms]).flatten()
        y = np.array([g.coords.xy[1] for g in difference.geoms]).flatten()
        i_intersect = get_index_closest_point(x, y, lon_p, lat_p)
        lon = x[i_intersect[0]]
        lat = y[i_intersect[0]]
    elif difference.geom_type == 'LineString': # if line is short, there is only one line
        lon = difference.coords[-1][0]
        lat = difference.coords[-1][1]
    else:
        raise ValueError(f'Difference did not return a MultiLineString or a LineString as expected: {difference.geom_type}')
        
    if lon == line.coords[-1][0] or lat == line.coords[-1][1]:
        # target point is just end point of line: did not cross target line
        warn(f'''No target point found, returning NaN values.
             If this happens for all points, consider increasing distance of initial line.''')
        lon = np.nan
        lat = np.nan
        
    return lon, lat

def find_transect_points_perpendicular_to_contours(lon_p:float, lat_p:float,
                                                   lon_p0:float, lat_p0:float,
                                                   lon_p2:float, lat_p2:float,
                                                   contours:list[shapely.LineString],
                                                   ds:float,
                                                   config:Config,
                                                   ) -> tuple[np.ndarray[float], np.ndarray[float]]:
    '''Finds points of the transect on each contour line, running as perpendicular to the contour
    lines as possible.'''

    # angle perpendicular to starting point
    perp_angle = get_perpendicular_angle(lon_p0, lat_p0, lon_p2, lat_p2)
    
    # line perpendicular to starting point, extending in both directions
    line = create_line_at_angle_to_point(lon_p, lat_p, perp_angle, config)
    
    lons = [lon_p]
    lats = [lat_p]
    
    perp_angle_previous = perp_angle
    
    for j in range(1, len(contours)):
        # point on contour
        lon, lat = get_intersection_point_between_lines(line, contours[j], lons[j-1], lats[j-1])
        lons.append(lon)
        lats.append(lat)
        
        if np.isnan(lon) or np.isnan(lat):
            return np.nan, np.nan
            
        # points on contour just before and after point (needed to calculate angle)
        s = shapely.line_locate_point(contours[j], shapely.Point(lon, lat)) # distance of point along the contour line
        sstart = shapely.line_locate_point(contours[j], contours[j].boundary.geoms[0])
        send = shapely.line_locate_point(contours[j], contours[j].boundary.geoms[1])
        s0 = s - ds
        s1 = s + ds
        if np.round(s0, 3) >= np.round(sstart, 3):
            p0 = shapely.line_interpolate_point(contours[j], s0) 
        else:
            p0 = shapely.Point(lon, lat)
        if np.round(s1, 3) <= np.round(send, 3):
            p1 = shapely.line_interpolate_point(contours[j], s1)
        else:
            p1 = shapely.Point(lon, lat)
        
        # angle perpendicular to point on contour line
        perp_angle = get_perpendicular_angle(p0.x, p0.y, p1.x, p1.y)
        # smooth new angle if it deviates too much from previous one
        smoothed_angle = smooth_angle(perp_angle, perp_angle_previous, config.angle_tolerance)

        # perpendicular line from contour line
        line = create_line_at_angle_to_point(lon, lat, smoothed_angle, config)

        perp_angle_previous = smoothed_angle # update previous angle

    return np.array(lons), np.array(lats)

def find_transects_from_starting_points(ds_grid:xr.Dataset, contours:list[shapely.LineString],
                                        lon_ps:np.ndarray[float], lat_ps:np.ndarray[float],
                                        ds:float, config:Config, start_index=0):
    transects = {}
    
    for i in range(len(lon_ps)):
        # --- determine transect points on contour lines
        if i == 0:
            lon_p0 = lon_ps[i]
            lat_p0 = lat_ps[i]
            lon_p2 = lat_ps[i+1]
            lat_p2 = lat_ps[i+1]
        elif i == len(lon_ps)-1:
            lon_p0 = lon_ps[i-1]
            lat_p0 = lat_ps[i-1]
            lon_p2 = lon_ps[i]
            lat_p2 = lat_ps[i]
        else:
            lon_p0 = lon_ps[i-1]
            lat_p0 = lat_ps[i-1]
            lon_p2 = lon_ps[i+1]
            lat_p2 = lat_ps[i+1]
        
        lons_org, lats_org = find_transect_points_perpendicular_to_contours(
            lon_ps[i], lat_ps[i], lon_p0, lat_p0, lon_p2, lat_p2,
            contours, ds, config
        )

        if np.any(np.isnan(lons_org)) or np.any(np.isnan(lats_org)):
            log.info(f'Transect {i} not found for shelf point: {lon_ps[i], lat_ps[i]}')
            continue
        
        # --- get transect points in ocean model
        # interpolate along transect line to higher resolution points
        transect = shapely.LineString(list(zip(lons_org, lats_org)))
        point_distances = np.arange(0, transect.length, ds/4)
        transect_interp = shapely.LineString([transect.interpolate(distance) for distance in point_distances] + [transect.boundary.geoms[1]])
        x, y = transect_interp.coords.xy
        
        # get eta, xi indices in ocean model
        eta_all, xi_all = get_eta_xi_of_lon_lat_point(ds_grid.lon_rho.values, ds_grid.lat_rho.values, x, y)
        coords = list(zip(eta_all, xi_all))
        unique_coords = list(dict.fromkeys(coords))
        unique_coords_list = list(zip(*unique_coords))
        etas = unique_coords_list[0]
        xis = unique_coords_list[1]
        
        # get lon, lat in ocean model
        lons = ds_grid.lon_rho.values[etas, xis]
        lats = ds_grid.lat_rho.values[etas, xis]
        
        transects[f't{start_index+i}'] = {'lon_org': list(lons_org), 'lat_org': list(lats_org),
                              'eta': [int(eta) for eta in etas], 'xi': [int(xi) for xi in xis],
                              'lon': list(lons), 'lat': list(lats)}
        
    return transects

def generate_transects_json_file(ds_grid:xr.Dataset, config:Config, output_path:str):
    '''Creates transects and saves them to a json file.
    
    Transects are created by:
    1. determining equally spaced points on the shallowest contour line
    2. determining the angle (bearing) of the contour in this point
    3. finding the point perpendicular to the shelf on the next depth contour
    4. determining the angle of this point on the depth contour
    5. finding the next point perperdicular to the depth contour on the next depth contour
    6. iterating steps 4-5 for all depth contours
    
    These points are stored as lon_org and lat_org in the transects dictionary and json file.
    
    The closest points in the ocean model along each transect are then determined, and the
    eta, xi coordinates and corresponding lon, lat from the ocean model are stored in the
    transects dictionary and json file.
    
    This is done by:
    1. interpolating the transect found in the steps above to a higher resolution
    2. finding the unique eta, xi indices of the ocean model on these points
    3. determing the lon, lat values in the ocean model of these points
    
    Input:
    - ds: xarray.Dataset containing ocean model grid data
    - config: Config object
    - output_path: string with filepath to write to
      
    Output:
    json file with saved transects'''
    
    # get approximate grid resolution (in degrees):
    dx = np.nanmean(np.unique(np.diff(ds_grid.lon_rho.values, axis=1)))
    dy = np.nanmean(np.unique(np.diff(ds_grid.lat_rho.values, axis=0)))
    ds = np.nanmin([dx, dy])

    # get depth contours
    contours = get_depth_contours(ds_grid.lon_rho.values, ds_grid.lat_rho.values, ds_grid.h.values, config.transect_contours)
    # get starting points along shallowest depth contour
    lon_ps, lat_ps = get_starting_points(contours[0], ds)
        
    # create transects from starting points
    transects = find_transects_from_starting_points(ds_grid, contours, lon_ps, lat_ps, ds, config)

    log.info(f'Writing transects to json file: {output_path}')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transects, f, ensure_ascii=False, indent=4)
        
    return transects

def read_transects_dict_from_json(transects_file:str) -> dict:
    '''Reads all transects from json file and returns a
    dictionary of transects.'''

    with open(transects_file, 'r') as f:
        all_transects = json.load(f)
    transect_names = list(all_transects.keys())
    
    transects = {}
    for i in range(len(transect_names)):
        lon_org = all_transects[transect_names[i]]['lon_org']
        lat_org = all_transects[transect_names[i]]['lat_org']
        eta = all_transects[transect_names[i]]['eta']
        xi = all_transects[transect_names[i]]['xi']
        lon = all_transects[transect_names[i]]['lon']
        lat = all_transects[transect_names[i]]['lat']
        
        transects[transect_names[i]] = {'lon_org': lon_org, 'lat_org': lat_org,
                                        'eta': eta, 'xi': xi,
                                        'lon': lon, 'lat': lat}
        
    return transects

def read_transects_in_lon_lat_range_from_json(transects_file:str,
                                             lon_range:list[float],
                                             lat_range:list[float]) -> dict:
    '''Reads transects from json file and returns a dictionary with only
    transects within the requested longitude and latitude range. If 
    lon_range and lat_range are set to None, returns all transects
    in json file.
    
    Input:
    - transects_file: string with transect file name
    - lon_range: list with longitude range
    - lat_range: list with latitude range
    
    Output:
    - transects: dictionary with transects within range'''
    
    all_transects = read_transects_dict_from_json(transects_file)
    
    if lon_range is None and lat_range is None:
        return all_transects

    transects = {}
    for t in list(all_transects.keys()):
        l_lon = np.all(np.logical_and(lon_range[0] <= np.array(all_transects[t]['lon']),
                                      np.array(all_transects[t]['lon']) <= lon_range[1]))
        l_lat = np.all(np.logical_and(lat_range[0] <= np.array(all_transects[t]['lat']),
                                      np.array(all_transects[t]['lat']) <= lat_range[1]))
        
        l_range = np.logical_and(l_lon, l_lat)
        
        if l_range == True:
            transects[t] = all_transects[t]
        else:
            continue
    
    return transects
        
if __name__ == '__main__':
    model_input_dir = get_dir_from_json('cwa')
    grid_file = f'{model_input_dir}grid.nc'
    grid_ds = xr.open_dataset(grid_file)
    
    config = read_config('cwa')

    generate_transects_json_file(grid_ds, config, 'input/transects/cwa_transects.json')