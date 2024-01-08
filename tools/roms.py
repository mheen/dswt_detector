from tools.coordinates import get_points_on_line_between_points, get_distance_between_points
from matplotlib import path
import numpy as np
from scipy import spatial

def get_z(Vtransform:int,
          s:np.ndarray,
          h:np.ndarray,
          cs_r:np.ndarray,
          hc:np.ndarray) -> np.ndarray:
    '''Gets depth of ROMS sigma layers.
    
    Input parameters:
    Vtransform: vertical terrain-following transformation equation
    s: sigma layers [s] (using "s_rho" in ROMS, but "s_w" is also an option)
    h: bottom depths [eta, xi] ("h" in ROMS)
    cs_r: s-level stretching curve [s] ("Cs_r" in ROMS)
    hc: critical depth ("hc" in ROMS)
    '''

    output_shape = (len(cs_r),) + h.shape

    if Vtransform == 2:
        n = hc*s[:, None] + np.outer(cs_r, h)
        d = (1.0 + hc/h)
    else:
        raise ValueError(f'Depth conversion not implemented for Vtransform values other than 2: Vtransform={Vtransform}')

    z = n.reshape(output_shape)/d

    return z

def find_eta_xi_covering_lon_lat_box(lon:np.ndarray, lat:np.ndarray,
                                     lon_range:list[float],
                                     lat_range:list[float]) -> tuple[int]:
    '''Returns eta and xi coordinates that completely cover the specified
    longitude and latitude bounding box.
    --- Input ---
    lon: 2d array with longitude coordinates
    lat: 2d array with latitude coordinates
    lon_range: list with two corner longitudes
    lat_range: list with two corner latitudes
    --- Output ---
    (xi0, xi1, eta0, eta1): eta and xi index values covering the box
    
    Based on code from Rich Signell: 
    https://gis.stackexchange.com/questions/71630/subsetting-a-curvilinear-netcdf-file-roms-model-output-using-a-lon-lat-boundin
    '''
    
    bounding_box = np.array([lon_range[0], lon_range[1], lat_range[0], lat_range[1]])
    box_path = np.array([bounding_box[[0,1,1,0]], bounding_box[[2,2,3,3]]]).T
    p = path.Path(box_path)
    points = np.vstack((lon.flatten(), lat.flatten())).T
    n, m = np.shape(lon)
    inside = p.contains_points(points).reshape((n, m))
    xi, eta = np.meshgrid(range(m), range(n))
    xi0 = min(xi[inside])
    xi1 = max(xi[inside])
    eta0 = min(eta[inside])
    eta1 = max(eta[inside])
    
    return xi0, xi1, eta0, eta1

def get_eta_xi_of_lon_lat_point(lon:np.ndarray, lat:np.ndarray,
                                lon_p:np.ndarray, lat_p:np.ndarray) -> tuple:
        
        grid_coords_1d = list(zip(np.ravel(lon), np.ravel(lat)))
        kdtree = spatial.KDTree(grid_coords_1d)
        
        float_types = [float, np.float64]
        if type(lon_p) in float_types and type(lat_p) in float_types:
            distance, index = kdtree.query([lon_p, lat_p])
            eta, xi = np.unravel_index(index, lon.shape)
            return xi, eta
        etas = []
        xis = []
        for i in range(len(lon_p)):
            distance, index = kdtree.query([lon_p[i], lat_p[i]])
            eta, xi = np.unravel_index(index, lon.shape)
            etas.append(eta)
            xis.append(xi)
        return np.array(etas), np.array(xis)

def get_eta_xi_along_transect(lon:np.ndarray, lat:np.ndarray,
                              lon1:float, lat1:float,
                              lon2:float, lat2:float, ds:float) -> tuple:
    lons, lats = get_points_on_line_between_points(lon1, lat1, lon2, lat2, ds)
    etas, xis = get_eta_xi_of_lon_lat_point(lon, lat, lons, lats)

    coords = list(zip(etas, xis))
    unique_coords = list(dict.fromkeys(coords))
    unique_coords_list = list(zip(*unique_coords))
    eta_unique = unique_coords_list[0]
    xi_unique = unique_coords_list[1]
    
    return np.array(eta_unique), np.array(xi_unique)

def get_distance_along_transect(lons:np.ndarray, lats:np.ndarray):
    distance = [0]
    
    for i in range(len(lons)-1):
        d = get_distance_between_points(lons[i], lats[i], lons[i+1], lats[i+1])
        distance.append(d)
    distance = np.array(distance)
    
    return np.cumsum(distance) # distance in meters

def convert_roms_u_v_to_u_east_v_north(u:np.ndarray, v:np.ndarray, angle:np.ndarray) -> tuple:
    '''Convert u and v from curvilinear ROMS output to u eastwards and v northwards.
    This is done by:
    1. Converting u and v so that they are on rho-coordinate point (cell center).
    2. Rotating u and v so they are directed eastwards and northwards respectively.
    
    Example grid: "." = rho-point, "x" = u-point, "*" = v-point
         _________ _________ _________ _________
        |         |         |         |         |
        |    .    x    .    x    .    x    .    |
        |   0,2  0,2  1,2  1,2  2,2  2,2  3,2   |
        |         |         |         |         |
        |____*____|____*____|____*____|____*____|
        |   0,1   |   1,1   |   2,1   |   3,1   |
        |    .    x    .    x    .    x    .    |
        |   0,1  0,1  1,1  1,1  2,1  2,1  3,1   |
        |         |         |         |         |
        |____*____|____*____|____*____|____*____|
        |   0,0   |   1,0   |   2,0   |   3,0   |
        |    .    x    .    x    .    x    .    |
        |   0,0  0,0  1,0  1,0  2,0  2,0  3,0   |
     ^  |         |         |         |         |
     |  |_________|_________|_________|_________|
    eta  xi ->
    '''

    def u2rho(var_u:np.ndarray) -> np.ndarray:
        '''Convert variable on u-coordinate to rho-coordinate.'''
        var_u_size = var_u.shape
        n_dimension = len(var_u_size)
        if n_dimension == 4:
            T = var_u_size[0]
            S = var_u_size[1]
            M = var_u_size[-2]
            L = var_u_size[-1]
            var_rho = np.empty((T, S, M, L+1))*np.nan
            var_rho[:, :, :, 1:L] = 0.5*(var_u[:, :, :, 0:-1]+var_u[:, :, :, 1:]) # averages in middle of grid
            var_rho[:, :, :, 0] = var_u[:, :, :, 0] # single right value for first rho-point
            var_rho[:, :, :, -1] = var_u[:, :, :, -1] # single left value for last rho-point
        else:
            raise ValueError('Conversion from u- to rho-coordinate only implemented for 4D variables.')
            
        return var_rho

    def v2rho(var_v:np.ndarray) -> np.ndarray:
        '''Convert variable on v-coordinate to rho-coordinate.'''
        var_v_size = var_v.shape
        n_dimension = len(var_v_size)
        if n_dimension == 4:
            T = var_v_size[0]
            S = var_v_size[1]
            M = var_v_size[-2]
            L = var_v_size[-1]
            var_rho = np.empty((T, S, M+1, L))*np.nan
            var_rho[:, :, 1:M, :] = 0.5*(var_v[:, :, 0:-1, :]+var_v[:, :, 1:, :]) # averages in middle of grid
            var_rho[:, :, 0, :] = var_v[:, :, 0, :] # single bottom value for first rho-point
            var_rho[:, :, -1, :] = var_v[:, :, -1, :] # single top value for last rho-point
        return var_rho

    def rotate_u_v(u_rho:np.ndarray, v_rho:np.ndarray, angle:np.ndarray) -> tuple:
        '''Rotate u and v velocities on curvilinear grid so that
        they are directed east- and northwards respectively.'''
        u_east = u_rho*np.cos(angle)-v_rho*np.sin(angle)
        v_north = v_rho*np.cos(angle)+u_rho*np.sin(angle)
        return u_east, v_north

    u_rho = u2rho(u)
    v_rho = v2rho(v)
    u_east, v_north = rotate_u_v(u_rho, v_rho, angle)

    return u_east, v_north
