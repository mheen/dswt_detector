from tools.roms import get_eta_xi_along_transect, find_eta_xi_covering_lon_lat_box
from tools.coordinates import get_bearing_between_points

import numpy as np
import xarray as xr
import rasterio
from rasterio import features
import shapely
from shapely.ops import nearest_points
from pyproj import Geod
from skimage import measure
from warnings import warn

import matplotlib.pyplot as plt
from plot_tools.basic_maps import plot_basic_map, plot_contours
import cartopy.crs as ccrs

mask = np.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]) # small test

def convert_land_mask_to_polygons(lon:np.ndarray[float], lat:np.ndarray[float], mask:np.ndarray[int]) -> list[shapely.Polygon]:
    land_mask = np.abs(mask-1).astype(np.int32) # assuming ocean points are 1 in mask!
    
    # create transformation so that lon and lat coordinates are used to create polygons
    # !!! FIX !!! transformation probably not working correctly
    gcps = []
    for i in range(land_mask.shape[0]):
        for j in range(land_mask.shape[1]):
            gcps.append(rasterio.control.GroundControlPoint(row=i, col=j, x=lon[i, j], y=lat[i, j]))
    affine_transform = rasterio.transform.from_gcps(gcps)
    
    shapes = features.shapes(land_mask, mask=land_mask.astype(bool), transform=affine_transform) # mask so that only land polygons are returned
    
    land_polys = []
    for vec, _ in shapes:
        land_polys.append(shapely.geometry.shape(vec))
        
    return land_polys

def get_largest_land_polygon(land_polys:list[shapely.Polygon]) -> shapely.Polygon:
    geod = Geod(ellps='WGS84')
    
    areas = []
    for poly in land_polys:
        areas.append(abs(geod.geometry_area_perimeter(poly)[0]))
        
    i_max = np.where(np.array(areas) == max(areas))[0][0]
    
    return land_polys[i_max]

def get_continental_shelf_grid_indices(h:np.ndarray, shelf_depth=200) -> tuple[int, int]:
    # not using this: using grid points means angles are all square, gives weird transects
    contours = measure.find_contours(h, shelf_depth)
    contour_lengths = np.array([len(contour) for contour in contours])
    longest_contour = contours[np.where(contour_lengths == np.nanmax(contour_lengths))[0][0]]
    # longest_contour is an array with interpolated i, j indices to shelf_depth
    # I am ignoring the fraction of the indices and instead flooring them to the nearest full index
    # this means that the returned locations will not be exactly at the shelf_depth
    # but this is not a strict requirement to generate transects, and the shelf_depth is likely
    # chosen deeper than the actual depth that will be used to analyse for DSWT
    eta = np.floor(longest_contour[:, 0]).astype(int)
    xi = np.floor(longest_contour[:, 1]).astype(int)
    
    return (eta, xi)

def get_continental_shelf_points(lon:np.ndarray, lat:np.ndarray, h:np.ndarray,
                                 shelf_depth=200) -> tuple[float, float]:
    
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
    
    perpendicular_angle = angle+90 # !!! FIX !!! needs to be either +90 or -90 depending on direction
    
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
    
if __name__ == '__main__':    
    ds_full = xr.load_dataset('tests/data/ozroms_20170613.nc')
    # WA
    lon_range = [114.0, 116.0]
    lat_range = [-34.0, -31.0]
    # # SA
    # lon_range = [132.0, 140.0]
    # lat_range = [-39.0, -29.0]
    # # GSR
    # lon_range = None
    # lat_range = None
    
    if lon_range is not None and lat_range is not None:
        xi0, xi1, eta0, eta1 = find_eta_xi_covering_lon_lat_box(ds_full.lon_rho.values, ds_full.lat_rho.values, lon_range, lat_range)
        ds = ds_full.isel(eta_rho=slice(eta0, eta1), xi_rho=slice(xi0, xi1))
    else:
        ds = ds_full
    
    lon = ds.lon_rho.values
    lat = ds.lat_rho.values
    
    land_polys = convert_land_mask_to_polygons(lon, lat, ds.mask_rho.values)
    largest_land = get_largest_land_polygon(land_polys)
    
    lon_ps, lat_ps = get_continental_shelf_points(lon, lat, ds.h.values)
        
    angles = calculate_perpendicular_angles_to_shelf_points(lon_ps, lat_ps)    
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax = plot_basic_map(ax, lon_range=lon_range, lat_range=lat_range)
    ax = plot_contours(lon, lat, ds.h.values, ax=ax, clevels=[200])
    x, y = largest_land.exterior.xy
    ax.plot(x, y, transform=ccrs.PlateCarree())

    for i in range(len(lon_ps)):
        lon_land, lat_land = find_closest_land_point_at_angle(lon_ps[i], lat_ps[i], angles[i], largest_land, dist=0.001)
        if np.isnan(lon_land) or np.isnan(lat_land):
            ax.plot(lon_ps[i], lat_ps[i], 'xr')
            continue
        eta, xi = get_eta_xi_along_transect(ds.lon_rho.values, ds.lat_rho.values, lon_land, lat_land, lon_ps[i], lat_ps[i], 500.)
        ax.plot(lon[eta, xi], lat[eta, xi], '.')
        ax.plot(lon[eta, xi], lat[eta, xi], '-')
        ax.plot(lon_ps[i], lat_ps[i], 'xk')
        ax.plot(lon_land, lat_land, 'ok')

    plt.show()
