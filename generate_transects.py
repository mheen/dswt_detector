from tools.roms import get_eta_xi_along_transect
from tools.coordinates import get_bearing_between_points

import numpy as np
import xarray as xr
import rasterio
from rasterio import features
import shapely
from shapely.ops import nearest_points
from pyproj import Geod

import matplotlib.pyplot as plt
from plot_tools.basic_maps import plot_basic_map
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

def get_nearest_point_on_polygon(polygon:shapely.Polygon, lon_p:float, lat_p:float) -> tuple[float, float]:
    point = shapely.Point(lon_p, lat_p)
    p1, _ = nearest_points(polygon, point)
    
    return (p1.x, p1.y)

def get_nearest_bearing_to_polygon(polygon:shapely.Polygon, lon_p:float, lat_p:float) -> float:
    x, y = get_nearest_point_on_polygon(polygon, lon_p, lat_p)
    bearing = get_bearing_between_points(lon_p, lat_p, x, y)
    
    return bearing
    
if __name__ == '__main__':
    ds = xr.load_dataset('tests/data/ozroms_gsr_landmask.nc')
    lon = ds.lon_rho.values
    lat = ds.lat_rho.values
    
    lon_ps = [114.0, 115.3, 120.7, 140.9, 152.3]
    lat_ps = [-32.0, -33.2, -34.6, -40.2, -35.2]
    
    land_polys = convert_land_mask_to_polygons(ds.lon_rho.values, ds.lat_rho.values, ds.mask_rho.values)
    largest_land = get_largest_land_polygon(land_polys)
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax = plot_basic_map(ax)
    x, y = largest_land.exterior.xy
    ax.plot(x, y, transform=ccrs.PlateCarree())

    for i in range(len(lon_ps)):
        x, y = get_nearest_point_on_polygon(largest_land, lon_ps[i], lat_ps[i])
        eta, xi = get_eta_xi_along_transect(ds.lon_rho.values, ds.lat_rho.values, x, y, lon_ps[i], lat_ps[i], 500.)
        ax.plot(lon[eta, xi], lat[eta, xi], '.')
        ax.plot(lon[eta, xi], lat[eta, xi], '-')
        ax.plot(lon_ps[i], lat_ps[i], 'xr')
        ax.plot(x, y, 'or')

    plt.show()
