import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from readers.read_ocean_data import select_input_files, load_roms_data, select_roms_subset, read_roms_data, get_roms_contour_coordinates, get_roms_ds_along_contour
from readers.read_meteo_data import load_era5_data, select_era5_subset_along_coordinates
from tools.roms import get_eta_xi_of_lon_lat_point, convert_roms_u_v_to_u_east_v_north
from tools.velocity_shore_angles import get_cross_and_along_shelf_velocities
from tools.timeseries import get_l_time_range
from transects import read_transects_in_lon_lat_range_from_json
from tools.timeseries import get_monthly_means, get_time_index
from tools.coordinates import get_distance_between_points
from tools.buoyancy_flux import calculate_buoyancy_heat_flux, calculate_buoyancy_salt_flux
from tools.wind import convert_u_v_to_meteo_vel_dir
from tools import log
from tools.files import get_dir_from_json, create_dir_if_does_not_exist, get_files_in_dir

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import pandas as pd
import os

OMEGA = 7.292*10**-5 # rad/s
RHO0 = 1025.
KAPPA = 0.4
CD = 0.0025
LAT0 = -32.0
F = 2*OMEGA*np.sin(np.deg2rad(LAT0))
G = 9.81 # m2/s

RI_CRIT = 0.30 # critical value for local Richardson number to determine boundary layers

def convert_sustr_svstr_to_rho_east_north(roms_ds:xr.Dataset):
    sustr_eastward, svstr_northward = convert_roms_u_v_to_u_east_v_north(roms_ds.sustr.values, roms_ds.svstr.values, roms_ds.angle.values, roms_ds.mask_rho.values)
    roms_ds['sustr_eastward'] = (['ocean_time', 'eta_rho', 'xi_rho'], sustr_eastward)
    roms_ds['svstr_northward'] = (['ocean_time', 'eta_rho', 'xi_rho'], svstr_northward)
    return roms_ds

def calculate_surface_ekman_layer(ds_stress_contour:xr.Dataset):
    tau_s = np.sqrt(ds_stress_contour.sustr_eastward.values**2 + ds_stress_contour.svstr_northward.values**2)
    
    hes = KAPPA / F * np.sqrt(tau_s / RHO0)
    
    return hes
    
def calculate_bottom_ekman_layer(ds_roms_contour:xr.Dataset):
    tau_b = CD * (ds_roms_contour.u_eastward.values[0, :]**2 + ds_roms_contour.v_northward.values[0, :])
    
    heb = 1 / F * np.sqrt(tau_b / RHO0)
    
    return heb

def calculate_surface_bottom_ekman_transport(ds_stress_contour:xr.Dataset, ds_roms_contour:xr.Dataset, contour_length:float):
    # theoretical estimates of Ekman transport
    Tes = - ds_stress_contour.stress_along.values / (RHO0 * F)
    Tes = np.nansum(Tes * ds_roms_contour.dx.values) / contour_length
    
    Teb = - CD * ds_roms_contour.u_along.values[0, :]**2 / (RHO0 * F)
    Teb = np.nansum(Teb * ds_roms_contour.dx.values) / contour_length
    
    return Tes, Teb

def estimate_us_ub(ds_roms_contour:xr.Dataset, contour_length:float):
    '''Estimates cross-shelf transport in the surface and bottom layers
    along a contour based on when surface and bottom velocities
    change sign.'''
    n_z_layers = len(ds_roms_contour.s_rho)
    
    Uss = np.zeros(len(ds_roms_contour.distance))
    Usb = np.zeros(len(ds_roms_contour.distance))
    zss = np.zeros(len(ds_roms_contour.distance))
    zsb = np.zeros(len(ds_roms_contour.distance))
    
    for i in range(len(ds_roms_contour.distance)):
        surface_layer_extends_to_bottom = False
        
        u_cross = ds_roms_contour.u_cross.values[:, i]
        delta_z = ds_roms_contour.delta_z.values[:, i]
        # --- surface ---
        if u_cross[-1] < 0:
            k = (u_cross[::-1] > 0).argmax()
            if k == 0:
                if np.any(u_cross[::-1] > 0): # means transition is right at surface
                    k = n_z_layers - 1
                else: # no transition at all
                    k = 0
                    surface_layer_extends_to_bottom = True
            else:
                k = n_z_layers - k
                
        elif u_cross[-1] > 0:
            k = (u_cross[::-1] < 0).argmax()
            if k == 0:
                if np.any(u_cross[::-1] < 0): # means transition is right at surface
                    k = n_z_layers - 1
                else: # no transition at all
                    k = 0
                    surface_layer_extends_to_bottom = True
            else:
                k = n_z_layers - k
        elif np.any(np.isnan(u_cross)):
            continue
        else:
            raise ValueError(f'Surface cross-shelf velocity equals zero: {u_cross[-1]}')
        
        Uss[i] = np.nansum(u_cross[k:] * delta_z[k:])
        zss[i] = np.nansum(delta_z[k:])
        
        # --- bottom ---
        if surface_layer_extends_to_bottom == True:
            continue
        
        def _find_transition_bottom(start_k):
            if u_cross[start_k] < 0:
                k = (u_cross > 0).argmax()
                if k == 0: # transition right at bottom (otherwise should not have gotten here at all)
                    k = 1
            elif u_cross[start_k] > 0:
                k = (u_cross < 0).argmax()
                if k == 0: # transition right at bottom (otherwise should not have gotten here at all)
                    k = 1
            else:
                raise ValueError(f'Bottom cross-shelf velocity equals zero: {u_cross[0]}')
            return k
        
        start_k = 0
        if u_cross[start_k] == 0.0:
            start_k += 1
        kb = _find_transition_bottom(start_k)
        
        Usb[i] = np.nansum(u_cross[:kb] * delta_z[:kb])
        zsb[i] = np.nansum(delta_z[:kb])
        
    Uss_total = np.nansum(Uss * ds_roms_contour.dx.values) / contour_length
    Usb_total = np.nansum(Usb * ds_roms_contour.dx.values) / contour_length
    zss_mean = np.nanmean(zss)
    zsb_mean = np.nanmean(zsb)
    
    return Uss_total, Usb_total, zss_mean, zsb_mean

def calculate_surface_and_bottom_mld(ds_roms_contour:xr.Dataset):
    drho_s = ds_roms_contour.drho_z.values - np.repeat(np.expand_dims(ds_roms_contour.drho_z.values[-1, :], axis=0), len(ds_roms_contour.s_rho), axis=0)
    mld_condition_s = abs(drho_s) / RHO0 < 10**-4
    
    drho_b = ds_roms_contour.drho_z.values - np.repeat(np.expand_dims(ds_roms_contour.drho_z.values[0, :], axis=0), len(ds_roms_contour.s_rho), axis=0)
    mld_condition_b = abs(drho_b) / RHO0 < 10**-4
    
    # depth index for which MLD condition fails from the bottom
    # (argmax finds first instance when True (1) is returned)
    false_mld_condition_b = mld_condition_b == False
    i_bottom = false_mld_condition_b.argmax(axis=0)
    
    # bottom MLD is zero unless it is not
    mld_b = np.zeros(len(ds_roms_contour.distance))
    i_distance_b = np.where(i_bottom != 0)
    mld_b[i_distance_b] = ds_roms_contour.h.values[i_distance_b] + ds_roms_contour.z_rho.values[i_bottom[i_distance_b], i_distance_b]
    
    # depth index for which MLD condition fails from the surface
    # (argmax of flipped depth finds first instance from last index when True (1) is returned)
    # convert this back to non-flipped index using depth shape (len - index - 1)
    false_mld_condition_s = mld_condition_s == False
    i_surface = len(ds_roms_contour.s_rho) - false_mld_condition_s[::-1, :].argmax(axis=0) - 1
    
    # surface MLD extends up to the bottom unless it doesn't
    mld_s = -ds_roms_contour.h.values
    i_distance_s = np.where(i_surface != len(ds_roms_contour.s_rho) - 1)
    mld_s[i_distance_s] = ds_roms_contour.z_rho.values[i_surface[i_distance_s], i_distance_s]
    
    return np.nanmean(mld_s), np.nanmean(mld_b)

def calculate_gradient_richardson_number(ds_roms_contour:xr.Dataset):
    drho_dz = ds_roms_contour.drho_z.values / ds_roms_contour.delta_z.values
    
    def _convert_to_rho(values):
        values_rho = np.empty((values.shape[0]+1, values.shape[1])) * np.nan
        values_rho[1:-1, :] = 0.5 * (values[0:-1, :] + values[1:, :])
        values_rho[0, :] = values[0, :]
        values_rho[-1, :] = values[-1, :]
        return values_rho
    
    du = np.diff(ds_roms_contour.u_eastward.values, axis=0)
    du = _convert_to_rho(du)
    dv = np.diff(ds_roms_contour.v_northward.values, axis=0)
    dv = _convert_to_rho(dv)
    
    du_dz = du / ds_roms_contour.delta_z.values
    dv_dz = dv / ds_roms_contour.delta_z.values
    
    ri = (G * abs(drho_dz)) / (RHO0 * (du_dz**2 + dv_dz**2))
    
    return ri
    
def calculate_bulk_richardson_number(ds_roms_contour:xr.Dataset):
    # delta values are surface - bottom
    delta_rho = abs(ds_roms_contour.density.values[-1, :] - ds_roms_contour.density.values[0, :])
    delta_u = ds_roms_contour.u_eastward.values[-1, :] - ds_roms_contour.u_eastward.values[0, :]
    delta_v = ds_roms_contour.v_northward.values[-1, :] - ds_roms_contour.v_northward.values[0, :]

    ri_bulk = (G * delta_rho * ds_roms_contour.h.values) / (RHO0 * (delta_u**2 + delta_v**2))
    
    return np.nanmean(ri_bulk)

def calculate_buoyancy_fluxes(ds_flux, depth_range):
    
    l_depth = np.logical_and(ds_flux.h >= depth_range[0], ds_flux.h <= depth_range[1])
    
    shflux = np.nanmean(ds_flux.shflux.values[l_depth])
    ssflux = np.nanmean(ds_flux.ssflux.values[l_depth])
    sst = np.nanmean(ds_flux.temp_sur.values[l_depth])
    sss = np.nanmean(ds_flux.salt_sur.values[l_depth])
    
    bhf = calculate_buoyancy_heat_flux(shflux, sst)
    bwf = calculate_buoyancy_salt_flux(ssflux, sst)
    bf = bhf + bwf
    
    return sst, sss, shflux, ssflux, bhf, bwf, bf

def write_analysis_data_to_csv(model, years, model_input_dir, grid_file, wind_input_dir, output_dir,
                               lon_range, lat_range, depth_contour, depth_range_shallow, depth_range_deep):
    '''Writing twice-daily means to reflect the land-seabreeze signal dominant in summer months
       from Rafiq et al. (2020) making the split according to the following times:
       - seabreeze (southerly winds) from 09:00 until 23:00 Perth local time (01:00 to 15:00 UTC)
       - landbreeze (easterly winds) from 23:00 until 08:00 Perth local time (15:00 to 00:00 UTC)
    '''
    
    file_preface = f'{model}_'
    
    grid_ds = xr.load_dataset(grid_file)
    lon_contour, lat_contour, contour_length = get_roms_contour_coordinates(grid_ds, lon_range, lat_range, depth_contour)
    
    columns = ['time', 'wind_vel', 'wind_dir',
               'sst_sh', 'sss_sh', 'shflux_sh', 'ssflux_sh',
               'bhflux_sh', 'bwflux_sh', 'bflux_sh',
               'sst_dp', 'sss_dp', 'shflux_dp', 'ssflux_dp',
               'bhflux_dp', 'bwflux_dp', 'bflux_dp',
               'Tes', 'Teb', 'hes', 'heb',
               'mld_s', 'mld_b', 'ri_bulk',
               'Uss', 'Usb', 'zss', 'zsb']
    
    for year in years:
        input_dir = f'{model_input_dir}{year}/'
        
        output_path = f'{output_dir}analysis_{year}.csv'
        
        if os.path.exists(output_path):
            df_temp = pd.read_csv(output_path)
            time = df_temp['time'].values
            time_last = datetime.strptime(pd.unique(time)[-1], '%Y-%m-%d %H:%M:%S')
            if time_last == datetime(year, 12, 31, 15, 0):
                log.info(f'Output already exists for {year}, skipping.')
                continue
            else:
                log.info(f'''Output partially exists for {year}. Running from {time_last+timedelta(days=1)} onwards.
                        Please check to make sure that all transects for {time_last} were written to file.''')
                date_range = [time_last+timedelta(days=1), datetime(year, 12, 31)]
        else:
            date_range = [datetime(year, 1, 1), datetime(year, 12, 31)]
        
        # load wind data (yearly data)
        wind_ds = load_era5_data(wind_input_dir, str(year))
        times_wind = np.array([pd.to_datetime(d) for d in wind_ds.time.values])
        
        roms_files = select_input_files(input_dir, file_preface=file_preface, date_range=date_range)
        roms_files.sort()
        
        for file in roms_files:
            # Load ROMS data
            ds_roms = load_roms_data(file, grid_file)
            u_cross, u_along = get_cross_and_along_shelf_velocities(ds_roms.h.values, ds_roms.u_eastward.values, ds_roms.v_northward.values)
            # add cross and along-shelf velocity to ds_roms
            ds_roms['u_cross'] = (['ocean_time', 's_rho', 'eta_rho', 'xi_rho'], u_cross)
            ds_roms['u_along'] = (['ocean_time', 's_rho', 'eta_rho', 'xi_rho'], u_along)
            # extract data along contour only
            ds_roms_contour = get_roms_ds_along_contour(ds_roms, grid_ds, lon_contour, lat_contour)
            # get twice-daily mean data (reflecting land- and seabreeze times)
            times = np.array([pd.to_datetime(d) for d in ds_roms_contour.ocean_time.values])
            day = datetime(times[0].year, times[0].month, times[0].day, 0, 0)
            l_time_lb = get_l_time_range(times, day + timedelta(hours=15), day + timedelta(hours=24))
            l_time_sb = get_l_time_range(times, day + timedelta(hours=1), day + timedelta(hours=15))
            ds_roms_contour_lb = ds_roms_contour.sel(ocean_time=l_time_lb).mean(dim='ocean_time')
            ds_roms_contour_sb = ds_roms_contour.sel(ocean_time=l_time_sb).mean(dim='ocean_time')
            
            # Load surface stress data
            sflux_file = select_input_files(f'{input_dir}shflux/',
                                            file_preface=f'{file_preface}{day.strftime("%Y%m%d")}_')
            ds_roms_stress = read_roms_data(sflux_file[0], grid_file, None)
            ds_roms_stress = convert_sustr_svstr_to_rho_east_north(ds_roms_stress)
            stress_cross, stress_along = get_cross_and_along_shelf_velocities(ds_roms_stress.h.values, ds_roms_stress.sustr_eastward.values, ds_roms_stress.svstr_northward.values)
            # add cross and along-shelf wind stress to ds_roms_stress
            ds_roms_stress['stress_cross'] = (['ocean_time', 'eta_rho', 'xi_rho'], stress_cross)
            ds_roms_stress['stress_along'] = (['ocean_time', 'eta_rho', 'xi_rho'], stress_along)
            ds_roms_stress_contour = get_roms_ds_along_contour(ds_roms_stress, grid_ds, lon_contour, lat_contour)
            # get twice-daily mean data (reflecting land- and seabreeze times)
            times_stress = np.array([pd.to_datetime(d) for d in ds_roms_stress_contour.ocean_time.values])
            l_time_lb = get_l_time_range(times_stress, day + timedelta(hours=15), day + timedelta(hours=24))
            l_time_sb = get_l_time_range(times_stress, day + timedelta(hours=1), day + timedelta(hours=15))
            ds_roms_stress_lb = ds_roms_stress.sel(ocean_time=l_time_lb).mean(dim='ocean_time')
            ds_roms_stress_sb = ds_roms_stress.sel(ocean_time=l_time_sb).mean(dim='ocean_time')
            ds_roms_stress_contour_lb = ds_roms_stress_contour.sel(ocean_time=l_time_lb).mean(dim='ocean_time')
            ds_roms_stress_contour_sb = ds_roms_stress_contour.sel(ocean_time=l_time_sb).mean(dim='ocean_time')
            
            # get wind data
            wind_ds_contour = select_era5_subset_along_coordinates(wind_ds, lon_contour, lat_contour)
            l_time_lb = get_l_time_range(times_wind, day + timedelta(hours=15), day + timedelta(hours=24))
            l_time_sb = get_l_time_range(times_wind, day + timedelta(hours=1), day + timedelta(hours=15))
            wind_ds_contour_lb = wind_ds_contour.sel(time=l_time_lb).mean(dim='time')
            wind_ds_contour_sb = wind_ds_contour.sel(time=l_time_sb).mean(dim='time')
            wind_vel_lb, wind_dir_lb = convert_u_v_to_meteo_vel_dir(np.nanmean(wind_ds_contour_lb.Uwind.values), np.nanmean(wind_ds_contour_lb.Vwind.values))
            wind_vel_sb, wind_dir_sb = convert_u_v_to_meteo_vel_dir(np.nanmean(wind_ds_contour_sb.Uwind.values), np.nanmean(wind_ds_contour_sb.Vwind.values))
            
            # get buoyancy fluxes
            (sst_lb_sh,
             sss_lb_sh,
             shflux_lb_sh,
             ssflux_lb_sh,
             bhf_lb_sh,
             bwf_lb_sh,
             bf_lb_sh) = calculate_buoyancy_fluxes(ds_roms_stress_lb, depth_range_shallow)
            (sst_lb_dp,
             sss_lb_dp,
             shflux_lb_dp,
             ssflux_lb_dp,
             bhf_lb_dp,
             bwf_lb_dp,
             bf_lb_dp) = calculate_buoyancy_fluxes(ds_roms_stress_lb, depth_range_deep)
            (sst_sb_sh,
             sss_sb_sh,
             shflux_sb_sh,
             ssflux_sb_sh,
             bhf_sb_sh,
             bwf_sb_sh,
             bf_sb_sh) = calculate_buoyancy_fluxes(ds_roms_stress_sb, depth_range_shallow)
            (sst_sb_dp,
             sss_sb_dp,
             shflux_sb_dp,
             ssflux_sb_dp,
             bhf_sb_dp,
             bwf_sb_dp,
             bf_sb_dp) = calculate_buoyancy_fluxes(ds_roms_stress_sb, depth_range_deep)
            
            # calculate layer depths
            hes_lb = calculate_surface_ekman_layer(ds_roms_stress_contour_lb)
            hes_sb = calculate_surface_ekman_layer(ds_roms_stress_contour_sb)
            heb_lb = calculate_bottom_ekman_layer(ds_roms_contour_lb)
            heb_sb = calculate_bottom_ekman_layer(ds_roms_contour_sb)
            mld_s_lb, mld_b_lb = calculate_surface_and_bottom_mld(ds_roms_contour_lb)
            mld_s_sb, mld_b_sb = calculate_surface_and_bottom_mld(ds_roms_contour_sb)
            ri_lb = calculate_bulk_richardson_number(ds_roms_contour_lb)
            ri_sb = calculate_bulk_richardson_number(ds_roms_contour_sb)
            
            # calculate cross-shelf Ekman transport
            Tes_lb, Teb_lb = calculate_surface_bottom_ekman_transport(ds_roms_stress_contour_lb, ds_roms_contour_lb, contour_length)
            Tes_sb, Teb_sb = calculate_surface_bottom_ekman_transport(ds_roms_stress_contour_sb, ds_roms_contour_sb, contour_length)
            
            # calculate cross-shelf transport
            Uss_lb, Usb_lb, zss_lb, zsb_lb = estimate_us_ub(ds_roms_contour_lb, contour_length)
            Uss_sb, Usb_sb, zss_sb, zsb_sb = estimate_us_ub(ds_roms_contour_sb, contour_length)
            
            # get data columns
            data_lb = [
                day + timedelta(hours=15),
                wind_vel_lb, wind_dir_lb,
                sst_lb_sh, sss_lb_sh, shflux_lb_sh, ssflux_lb_sh,
                bhf_lb_sh, bwf_lb_sh, bf_lb_sh,
                sst_lb_dp, sss_lb_dp, shflux_lb_dp, ssflux_lb_dp,
                bhf_lb_dp, bwf_lb_dp, bf_lb_dp,
                Tes_lb, Teb_lb,
                np.nanmean(hes_lb), np.nanmean(heb_lb),
                mld_s_lb, mld_b_lb, ri_lb,
                Uss_lb, Usb_lb, zss_lb, zsb_lb
            ]
            
            data_sb = [
                day + timedelta(hours=1),
                wind_vel_sb, wind_dir_sb,
                sst_sb_sh, sss_sb_sh, shflux_sb_sh, ssflux_sb_sh,
                bhf_sb_sh, bwf_sb_sh, bf_sb_sh,
                sst_sb_dp, sss_sb_dp, shflux_sb_dp, ssflux_sb_dp,
                bhf_sb_dp, bwf_sb_dp, bf_sb_dp,
                Tes_sb, Teb_sb,
                np.nanmean(hes_sb), np.nanmean(heb_sb),
                mld_s_sb, mld_b_sb, ri_sb,
                Uss_sb, Usb_sb, zss_sb, zsb_sb
            ]
            
            df = pd.DataFrame(data=[data_sb, data_lb], columns=columns)
            log.info(f'Writing analysis data to file: {output_path}')
            if os.path.exists(output_path):
                df.to_csv(output_path, mode='a', header=False, index=False)
            else:
                df.to_csv(output_path, index=False)

def write_cross_shelf_transport_to_netcdf(model, year, model_input_dir, grid_file, output_path,
                                         lon_range, lat_range):
    file_preface = f'{model}_'
    
    grid_ds = xr.load_dataset(grid_file)
    grid_ds = select_roms_subset(grid_ds, time_range=None, lon_range=lon_range, lat_range=lat_range)
    
    input_dir = f'{model_input_dir}{year}/'
    
    date_range = [datetime(year, 1, 1), datetime(year, 12, 31)]
    
    roms_files = select_input_files(input_dir, file_preface=file_preface, date_range=date_range)
    roms_files.sort()
    
    n_days = (datetime(year+1, 1, 1) - datetime(year, 1, 1)).days
    time = []
    for n in range(n_days):
        time.append(datetime(year, 1, 1) + timedelta(days=n))
    time = np.array(time)
    
    data_var_3d = np.empty((len(time), len(grid_ds.eta_rho), len(grid_ds.xi_rho))) * np.nan
    data_var_4d = np.empty((len(time), len(grid_ds.s_rho), len(grid_ds.eta_rho), len(grid_ds.xi_rho))) * np.nan
    
    ds_cross = xr.Dataset(
        data_vars={
            "u_cross": (("ocean_time", "s_rho", "eta_rho", "xi_rho"), np.copy(data_var_4d)),
            "v_along": (("ocean_time", "s_rho", "eta_rho", "xi_rho"), np.copy(data_var_4d)),
            "u_bar": (("ocean_time", "eta_rho", "xi_rho"), np.copy(data_var_3d)),
            "u_prime": (("ocean_time", "s_rho", "eta_rho", "xi_rho"), np.copy(data_var_4d)),
            "L_vc": (("ocean_time", "eta_rho", "xi_rho"), np.copy(data_var_3d)),
            "u_cross_lb": (("ocean_time", "s_rho", "eta_rho", "xi_rho"), np.copy(data_var_4d)),
            "v_along_lb": (("ocean_time", "s_rho", "eta_rho", "xi_rho"), np.copy(data_var_4d)),
            "u_bar_lb": (("ocean_time", "eta_rho", "xi_rho"), np.copy(data_var_3d)),
            "u_prime_lb": (("ocean_time", "s_rho", "eta_rho", "xi_rho"), np.copy(data_var_4d)),
            "L_vc_lb": (("ocean_time", "eta_rho", "xi_rho"), np.copy(data_var_3d)),
            "u_cross_sb": (("ocean_time", "s_rho", "eta_rho", "xi_rho"), np.copy(data_var_4d)),
            "v_along_sb": (("ocean_time", "s_rho", "eta_rho", "xi_rho"), np.copy(data_var_4d)),
            "u_bar_sb": (("ocean_time", "eta_rho", "xi_rho"), np.copy(data_var_3d)),
            "u_prime_sb": (("ocean_time", "s_rho", "eta_rho", "xi_rho"), np.copy(data_var_4d)),
            "L_vc_sb": (("ocean_time", "eta_rho", "xi_rho"), np.copy(data_var_3d)),
            "lon_rho": (("eta_rho", "xi_rho"), grid_ds.lon_rho.values),
            "lat_rho": (("eta_rho", "xi_rho"), grid_ds.lat_rho.values),
            "h": (("eta_rho", "xi_rho"), grid_ds.h.values),
            "z_rho": (("s_rho", "eta_rho", "xi_rho"), np.empty((len(grid_ds.s_rho), len(grid_ds.eta_rho), len(grid_ds.xi_rho))) * np.nan)
                },
        coords={
            "ocean_time": time,
            "s_rho": grid_ds.s_rho.values,
            "eta_rho": grid_ds.eta_rho.values,
            "xi_rho": grid_ds.xi_rho.values
            }
        )
    
    for i, file in enumerate(roms_files):
        # Load ROMS data
        ds_roms = load_roms_data(file, grid_file)
        ds_roms = select_roms_subset(ds_roms, time_range=None, lon_range=lon_range, lat_range=lat_range)
        u_cross, u_along = get_cross_and_along_shelf_velocities(ds_roms.h.values, ds_roms.u_eastward.values, ds_roms.v_northward.values)
        # add cross and along-shelf velocity to ds_roms
        ds_roms['u_cross'] = (['ocean_time', 's_rho', 'eta_rho', 'xi_rho'], u_cross)
        ds_roms['u_along'] = (['ocean_time', 's_rho', 'eta_rho', 'xi_rho'], u_along)
        # add depth-mean cross-shelf velocity and u_cross'
        delta_z_4d = np.repeat(np.expand_dims(ds_roms.delta_z.values, axis=0), len(ds_roms.ocean_time), axis=0)
        h_3d = np.repeat(np.expand_dims(ds_roms.h.values, axis=0), len(ds_roms.ocean_time), axis=0)
        u_cross_bar = 1/h_3d * np.nansum(u_cross * delta_z_4d, axis=1)
        u_cross_prime = ds_roms.u_cross.values - np.repeat(np.expand_dims(u_cross_bar, axis=1), len(ds_roms.s_rho), axis=1)
        ds_roms['u_cross_bar'] = (['ocean_time', 'eta_rho', 'xi_rho'], u_cross_bar)
        ds_roms['u_cross_prime'] = (['ocean_time', 's_rho', 'eta_rho', 'xi_rho'], u_cross_prime)
        # add depth-integrated absolute value of u_cross' - L_vc
        L_vc = 1/h_3d * np.nansum(delta_z_4d * abs(u_cross_prime), axis=1)
        ds_roms['L_vc'] = (['ocean_time', 'eta_rho', 'xi_rho'], L_vc)
        
        # get twice-daily mean data (reflecting land- and seabreeze times) as well as daily-mean
        times = np.array([pd.to_datetime(d) for d in ds_roms.ocean_time.values])
        day = datetime(times[0].year, times[0].month, times[0].day, 0, 0)
        l_time_lb = get_l_time_range(times, day + timedelta(hours=15), day + timedelta(hours=24))
        l_time_sb = get_l_time_range(times, day + timedelta(hours=1), day + timedelta(hours=15))
        ds_roms_lb = ds_roms.sel(ocean_time=l_time_lb).mean(dim='ocean_time')
        ds_roms_sb = ds_roms.sel(ocean_time=l_time_sb).mean(dim='ocean_time')
        ds_roms_d = ds_roms.mean(dim='ocean_time')
        
        roms_file_time = datetime.strptime(os.path.basename(file)[len(file_preface):][:8], '%Y%m%d')
        i_time = get_time_index(time, roms_file_time)
        
        # add data to new netcdf
        if i == 0:
            ds_cross['z_rho'].values = ds_roms['z_rho'].values
        
        ds_cross['u_cross'].values[i_time, :, :, :] = ds_roms_d['u_cross'].values
        ds_cross['v_along'].values[i_time, :, :, :] = ds_roms_d['u_along'].values
        ds_cross['u_bar'].values[i_time, :, :] = ds_roms_d['u_cross_bar'].values
        ds_cross['u_prime'].values[i_time, :, :, :] = ds_roms_d['u_cross_prime'].values
        ds_cross['L_vc'].values[i_time, :, :] = ds_roms_d['L_vc'].values
        
        ds_cross['u_cross_lb'].values[i_time, :, :, :] = ds_roms_lb['u_cross'].values
        ds_cross['v_along_lb'].values[i_time, :, :, :] = ds_roms_lb['u_along'].values
        ds_cross['u_bar_lb'].values[i_time, :, :] = ds_roms_lb['u_cross_bar'].values
        ds_cross['u_prime_lb'].values[i_time, :, :, :] = ds_roms_lb['u_cross_prime'].values
        ds_cross['L_vc_lb'].values[i_time, :, :] = ds_roms_lb['L_vc'].values

        ds_cross['u_cross_sb'].values[i_time, :, :, :] = ds_roms_sb['u_cross'].values
        ds_cross['v_along_sb'].values[i_time, :, :, :] = ds_roms_sb['u_along'].values
        ds_cross['u_bar_sb'].values[i_time, :, :] = ds_roms_sb['u_cross_bar'].values
        ds_cross['u_prime_sb'].values[i_time, :, :, :] = ds_roms_sb['u_cross_prime'].values
        ds_cross['L_vc_sb'].values[i_time, :, :] = ds_roms_sb['L_vc'].values
        
    ds_cross.to_netcdf(output_path)
            
if __name__ == '__main__':
    
    model = 'cwa'
    
    model_input_dir = get_dir_from_json(model)
    grid_file = f'{model_input_dir}grid.nc'
    
    output_dir = get_dir_from_json('analysis')
    create_dir_if_does_not_exist(output_dir)
    
    wind_input_dir = get_dir_from_json('era5')
    
    lon_range = [114.0, 116.0]
    lat_range = [-33.0, -31.0]
    
    years = np.arange(2000, 2024)
    
    depth_contour = 50.0
    depth_range_shallow = [0, 20]
    depth_range_deep = [50, 300] # LC
    
    write_cross_shelf_transport_to_netcdf(model, 2017, model_input_dir, grid_file, f'{output_dir}/cross-shelf/ucross_2017.nc',
                                     lon_range, lat_range)
    
    write_analysis_data_to_csv(model, years, model_input_dir, grid_file, wind_input_dir, output_dir,
                               lon_range, lat_range, depth_contour, depth_range_shallow, depth_range_deep)
