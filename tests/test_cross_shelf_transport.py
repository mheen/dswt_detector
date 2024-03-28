import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

import pytest
import xarray as xr
import numpy as np
from datetime import datetime

from tools.config import read_config
from dswt.cross_shelf_transport import calculate_down_transect_velocity_component, calculate_dswt_cross_shelf_transport_along_transect

config = read_config('cwa')

distance = np.arange(0.0, 7.0)
h = np.array([40, 40, 40, 50, 50, 60, 70])
s_rho = np.arange(0, 5)
ocean_time = [datetime(2024, 3, 28)]
n_z_layers = len(s_rho)
z_rho = np.array([np.arange(-40, 0, 40/n_z_layers)+40/(2*n_z_layers),
                 np.arange(-40, 0, 40/n_z_layers)+40/(2*n_z_layers),
                 np.arange(-40, 0, 40/n_z_layers)+40/(2*n_z_layers),
                 np.arange(-50, 0, 50/n_z_layers)+50/(2*n_z_layers),
                 np.arange(-50, 0, 50/n_z_layers)+50/(2*n_z_layers),
                 np.arange(-60, 0, 60/n_z_layers)+60/(2*n_z_layers),
                 np.arange(-70, 0, 70/n_z_layers)+70/(2*n_z_layers)]).transpose()     
delta_z = np.diff(np.concatenate((z_rho, np.expand_dims(np.zeros(z_rho.shape[1]), axis=0))), axis=0)
dt = 3*60*60

lon_rho = np.arange(114.0, 116.0, 2/len(distance))
lat_rho = np.ones(len(distance))*-32.0

u_eastward = np.zeros(z_rho.shape)
u_eastward[0, 3] = 0.1
u_eastward[1, 3] = 0.2
u_eastward[2, 3] = 0.1
u_eastward[0, 4] = 0.1
u_eastward[1, 4] = 0.2
u_eastward[2, 4] = -0.1
u_eastward[0, 5] = 0.05
u_eastward[1, 5] = 0.1
u_eastward[2, 5] = 0.2
u_eastward = np.expand_dims(u_eastward, axis=0)

v_northward = np.expand_dims(np.zeros(z_rho.shape), axis=0)

vertical_density_gradient = np.zeros(z_rho.shape)
vertical_density_gradient[2, 3] = 0.02
vertical_density_gradient[2, 4] = 0.02
vertical_density_gradient[2, 5] = 0.02
vertical_density_gradient[1, 5] = 0.02
vertical_density_gradient = np.expand_dims(vertical_density_gradient, axis=0)

transect_ds = xr.Dataset(
    data_vars=dict(
        z_rho=(['s_rho', 'distance'], z_rho),
        delta_z=(['s_rho', 'distance'], delta_z),
        lon_rho=(['distance'], lon_rho),
        lat_rho=(['distance'], lat_rho),
        h=(['distance'], h),
        u_eastward=(['ocean_time', 's_rho', 'distance'], u_eastward),
        v_northward=(['ocean_time', 's_rho', 'distance'], v_northward),
        vertical_density_gradient=(['ocean_time', 's_rho', 'vertical_density_gradient'], vertical_density_gradient)
    ),
    coords=dict(
        distance=(['distance'], distance),
        ocean_time=(['ocean_time'], ocean_time),
        s_rho=(['s_rho'], s_rho)
    )
)

transect_ds['dt'] = dt

def test_down_transect_velocity():
    down_vel = calculate_down_transect_velocity_component(0.1, 0.0, 114.0, -32.0, 116.0, -32.0)
    assert np.round(down_vel, 1) == 0.1

def test_cross_shelf_transport():
    l_dswt = np.array([True])
    transport, mean_vel = calculate_dswt_cross_shelf_transport_along_transect(transect_ds, l_dswt, config)
    assert np.logical_and(np.round(mean_vel[0], 3) == 0.125, np.ceil(transport[0]) == 28080.0)
