import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.roms import get_distance_along_transect
from tools import log

from tools.timeseries import convert_datetime_to_time, get_l_time_range
from tools.seawater_density import calculate_density
import numpy as np
from datetime import datetime, timedelta
from scipy import interpolate
import xarray as xr
import pandas as pd
    
class GliderData:
    def __init__(self, time:np.ndarray,
                 lon:np.ndarray,
                 lat:np.ndarray,
                 depth:np.ndarray,
                 temp:np.ndarray,
                 salt:np.ndarray,
                 ox2:np.ndarray,
                 cphl:np.ndarray,
                 bbp:np.ndarray,
                 cdom:np.ndarray,
                 phase:np.ndarray):

        self.time = time
        self.lon = lon
        self.lat = lat
        self.depth = depth
        self.temp = temp
        self.salt = salt
        self.ox2 = ox2
        self.cphl = cphl
        self.bbp = bbp
        self.cdom = cdom
        self.phase = phase

        self.add_density()
        self.add_cumulative_time_along_glider_path()
        self.add_bottom()

    def add_density(self):
        self.density = calculate_density(self.salt, self.temp, self.depth)

    def add_cumulative_time_along_glider_path(self):
        self.cumtime, _ = convert_datetime_to_time(self.time, time_units='days', time_origin=self.time[0])

    def add_bottom(self) -> np.ndarray:
        '''Determines approximate ocean bottom by finding the bottom of each glider dive.'''
        
        bottom_times, bottom_depths = self.find_seafloor_from_dive_cycles()
        
        # Interpolate seafloor depth to grid time points
        seafloor_interp = interpolate.interp1d(
            bottom_times,
            bottom_depths,
            kind='linear',
            bounds_error=False,
            fill_value=(bottom_depths[0], bottom_depths[-1])
        )
        self.z_bottom = -seafloor_interp(self.cumtime)
        
    def find_seafloor_from_dive_cycles(self, window_size=11):
        l_nonan = ~np.isnan(self.depth)
        # Create DataFrame with your data
        df = pd.DataFrame({
            'time': self.cumtime[l_nonan],
            'depth': self.depth[l_nonan],
            'phase': self.phase[l_nonan],  # 0, 1, 4, 3 as you defined
            'lat': self.lat[l_nonan],
            'lon': self.lon[l_nonan]
        }).sort_values('time').reset_index(drop=True)
        
        # Identify start of each new dive (transition to phase 1)
        df['new_dive'] = (df['phase'] == 1) & (df['phase'].shift(1) != 1)
        df['dive_id'] = df['new_dive'].cumsum()

        # Get maximum depth for each complete dive cycle
        # Only use dives that have both descending (1) and ascending (4) phases
        dive_summary = []

        for dive_id, group in df.groupby('dive_id'):
            phases_in_dive = group['phase'].unique()

            # Check if dive is complete (has descent and ascent)
            has_descent = 1 in phases_in_dive
            has_ascent = 4 in phases_in_dive

            if has_descent and has_ascent:
                max_depth_idx = group['depth'].idxmax()
                dive_summary.append({
                    'dive_id': dive_id,
                    'time': group.loc[max_depth_idx, 'time'],
                    'depth': group['depth'].max(),
                    'n_points': len(group)
                })

        dive_df = pd.DataFrame(dive_summary)

        if len(dive_df) == 0:
            raise ValueError("No complete dive cycles found. Check phase data.")

        # Extract raw seafloor points
        bottom_times = dive_df['time'].values
        bottom_depths = dive_df['depth'].values

        # Ensure window_size is valid
        if window_size > len(bottom_depths):
            window_size = min(3, len(bottom_depths) if len(bottom_depths) % 2 == 1 else len(bottom_depths) - 1)
        if window_size % 2 == 0:
            window_size += 1  # Make odd

        # Apply simple moving average smoothing (centered)
        if window_size >= 3 and len(bottom_depths) >= window_size:
            # Pad edges to maintain length
            pad_size = window_size // 2
            padded = np.pad(bottom_depths, (pad_size, pad_size), mode='edge')
            smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')

            # Ensure same length (trim if needed)
            seafloor_depths_smooth = smoothed[:len(bottom_depths)]
        else:
            seafloor_depths_smooth = bottom_depths
        
        return bottom_times, seafloor_depths_smooth
    
    def get_gridded_transect_data(self, values, dt=1/24, dz=1, interpolated_data=True):
        # create grid along transect to interpolate to
        t = np.arange(np.nanmin(self.cumtime), np.nanmax(self.cumtime)+dt, dt)
        z = np.arange(np.nanmin(-self.depth), 0, dz)
        
        tt, zz = np.meshgrid(t, z)
        
        if np.all(np.isnan(values)):
            return t, z, np.empty(tt.shape) * np.nan
        
        l_nans = np.logical_or(np.isnan(self.depth), np.isnan(values))
        time_obs = self.cumtime[~l_nans]
        z_obs = -self.depth[~l_nans]
        values_obs = values[~l_nans]
        
        if interpolated_data == True:
            gridded_values = interpolate.griddata(points=(z_obs, time_obs),
                                                values=values_obs,
                                                xi=(zz, tt),
                                                method='linear')
            
            # remove values below seafloor
            bottom_times, bottom_depths = self.find_seafloor_from_dive_cycles()
            seafloor_interp = interpolate.interp1d(
                bottom_times,
                bottom_depths,
                kind='linear',
                bounds_error=False,
                fill_value=(bottom_depths[0], bottom_depths[-1])
            )
            z_bottom = -seafloor_interp(tt)
            l_below_seafloor = zz <= z_bottom
            gridded_values[l_below_seafloor] = np.nan
            
        else:
            values_sum, _, _ = np.histogram2d(z_obs, time_obs, bins=(z, t), weights=values_obs, density=False)
            n_values, _, _ = np.histogram2d(z_obs, time_obs, bins=(z, t))
            
            gridded_values = values_sum / n_values
            t = t[:-1] + np.diff(t)
            z = z[:-1] + np.diff(z)
        
        return t, z, gridded_values

    def get_data_in_time_frame(self, start_time:datetime, end_time:datetime):
        l_time = get_l_time_range(self.time, start_time, end_time)
        self.time = self.time[l_time]
        self.lon = self.lon[l_time]
        self.lat = self.lat[l_time]
        self.depth = self.depth[l_time]
        self.temp = self.temp[l_time]
        self.salt = self.salt[l_time]
        self.ox2 = self.ox2[l_time]
        self.cphl = self.cphl[l_time]
        self.bbp = self.bbp[l_time]
        self.cdom = self.cdom[l_time]
        self.phase = self.phase[l_time]
        self.cumtime = self.cumtime[l_time] - self.cumtime[l_time][0]
        self.z_bottom = self.z_bottom[l_time]
        self.density = self.density[l_time]

    @staticmethod
    def read_from_netcdf(input_path:str, use_qc_flags=[1, 2]):
        # IMOS standard quality control flags:
        # 0: no qc performed
        # 1: good data
        # 2: probably good data
        # 3: bad data that are potentially correctible
        # 4: bad data
        # 5: value changed
        # 6: not used
        # 7: interpolated values
        # 8: missing values
        
        ds = xr.load_dataset(input_path)
        
        time = np.array([pd.to_datetime(t) for t in ds.TIME.values])
        
        lon = ds.LONGITUDE.values
        l_lon = sum([ds.LONGITUDE_quality_control.values == qc_flag for qc_flag in use_qc_flags]).astype(bool)
        lon[~l_lon] = np.nan
        
        lat = ds.LATITUDE.values
        l_lat = sum([ds.LATITUDE_quality_control.values == qc_flag for qc_flag in use_qc_flags]).astype(bool)
        lat[~l_lat] = np.nan
        
        depth = ds.DEPTH.values
        l_depth = sum([ds.DEPTH_quality_control.values == qc_flag for qc_flag in use_qc_flags]).astype(bool)
        depth[~l_depth] = np.nan

        temp = ds.TEMP.values
        l_temp = sum([ds.TEMP_quality_control.values == qc_flag for qc_flag in use_qc_flags]).astype(bool)
        temp[~l_temp] = np.nan

        salt = ds.PSAL.values
        l_salt = sum([ds.PSAL_quality_control.values == qc_flag for qc_flag in use_qc_flags]).astype(bool)
        salt[~l_salt] = np.nan

        if 'DOX2' in list(ds.variables.keys()):
            ox2 = ds.DOX2.values
            l_ox2 = sum([ds.DOX2_quality_control.values == qc_flag for qc_flag in use_qc_flags]).astype(bool)
            ox2[~l_ox2] = np.nan
        else:
            ox2 = np.empty(time.shape) * np.nan

        if 'CPHL' in list(ds.variables.keys()):
            cphl = ds.CPHL.values
            l_cphl = sum([ds.CPHL_quality_control.values == qc_flag for qc_flag in use_qc_flags]).astype(bool)
            cphl[~l_cphl] = np.nan
        else:
            cphl = np.empty(time.shape) * np.nan

        if 'BBP' in list(ds.variables.keys()):
            bbp = ds.BBP.values
            l_bbp = sum([ds.BBP_quality_control.values == qc_flag for qc_flag in use_qc_flags]).astype(bool)
            bbp[~l_bbp] = np.nan
        else:
            bbp = np.empty(time.shape) * np.nan

        if 'CDOM' in list(ds.variables.keys()):
            cdom = ds.CDOM.values
            l_cdom = sum([ds.CDOM_quality_control.values == qc_flag for qc_flag in use_qc_flags]).astype(bool)
            cdom[~l_cdom] = np.nan
        else:
            cdom = np.empty(time.shape) * np.nan

        # phase (0: surface drifting, 1: descending profile, 4: ascending profile, 3: inflexion)
        phase = ds['PHASE'].values

        return GliderData(time, lon, lat, depth, temp, salt, ox2, cphl, bbp, cdom, phase)

def convert_glider_data_to_transect_data(glider_data:GliderData, output_path=None):
    time, z, temp = glider_data.get_gridded_transect_data(glider_data.temp)
    _, _, salt = glider_data.get_gridded_transect_data(glider_data.salt)
    _, _, density = glider_data.get_gridded_transect_data(glider_data.density)
    _, _, ox2 = glider_data.get_gridded_transect_data(glider_data.ox2)
    _, _, cphl = glider_data.get_gridded_transect_data(glider_data.cphl)
    _, _, bbp = glider_data.get_gridded_transect_data(glider_data.bbp)
    _, _, cdom = glider_data.get_gridded_transect_data(glider_data.cdom)
    
    l_loc = np.logical_and(~np.isnan(glider_data.lon), ~np.isnan(glider_data.lat))
    l_time = np.logical_and(time >= np.nanmin(glider_data.cumtime[l_loc]), time <= np.nanmax(glider_data.cumtime[l_loc]))
    
    f_lon = interpolate.interp1d(glider_data.cumtime[l_loc], glider_data.lon[l_loc])
    lon = f_lon(time[l_time])
    f_lat = interpolate.interp1d(glider_data.cumtime[l_loc], glider_data.lat[l_loc])
    lat = f_lat(time[l_time])
    f_h = interpolate.interp1d(glider_data.cumtime[~np.isnan(glider_data.z_bottom)], glider_data.z_bottom[~np.isnan(glider_data.z_bottom)])
    h = f_h(time[l_time])
    
    distance = get_distance_along_transect(lon, lat)
    
    dates = np.array([glider_data.time[0] + timedelta(days=t) for t in time[l_time]])
    
    transect_ds = xr.Dataset(
        data_vars={
            "temp": (("z", "distance"), temp[:, l_time]),
            "salt": (("z", "distance"), salt[:, l_time]),
            "density": (("z", "distance"), density[:, l_time]),
            "ox2": (("z", "distance"), ox2[:, l_time]),
            "cphl": (("z", "distance"), cphl[:, l_time]),
            "bbp": (("z", "distance"), bbp[:, l_time]),
            "cdom": (("z", "distance"), cdom[:, l_time]),
            "lon": (("distance"), lon),
            "lat": (("distance"), lat),
            "h": (("distance"), h),
            "time": (("distance"), dates)
        },
        coords={
            "distance": distance,
            "z": z
        }
    )
    
    if output_path is not None:
        log.info(f'Writing interpolated glider transect data to file: {output_path}')
        transect_ds.to_netcdf(output_path)
    
    return transect_ds

def get_glider_transect_data(glider_data:GliderData, flip=False):
    transect_ds = convert_glider_data_to_transect_data(glider_data)
    
    if flip == True:
        transect_ds.temp.values = np.fliplr(transect_ds.temp.values)
        transect_ds.salt.values = np.fliplr(transect_ds.salt.values)
        transect_ds.density.values = np.fliplr(transect_ds.density.values)
        transect_ds.ox2.values = np.fliplr(transect_ds.ox2.values)
        transect_ds.cphl.values = np.fliplr(transect_ds.cphl.values)
        transect_ds.bbp.values = np.fliplr(transect_ds.bbp.values)
        transect_ds.cdom.values = np.fliplr(transect_ds.cdom.values)
        transect_ds.h.values = np.flip(transect_ds.h.values)
    
    l_below_seafloor = transect_ds.z <= transect_ds.h
    
    if flip == True:
        dx = abs(np.diff(transect_ds.distance.values[::-1]))
    else:
        dx = np.diff(transect_ds.distance.values)
    # convert dx back to rho-points
    dx_rho = np.empty(transect_ds.distance.shape) * np.nan
    dx_rho[1:-1] = 0.5 * (dx[0:-1] + dx[1:])
    dx_rho[0] = dx[0]
    dx_rho[-1] = dx[-1]
    transect_ds['delta_x'] = (['distance'], dx_rho)
    
    # --- add depth mean density
    delta_z = np.diff(transect_ds.z)
    delta_z_rho = np.empty(transect_ds.z.shape) * np.nan
    delta_z_rho[1:-1] = 0.5 * (delta_z[0:-1] + delta_z[1:])
    delta_z_rho[0] = delta_z[0]
    delta_z_rho[-1] = delta_z[-1]
    delta_z_rho_2d = np.repeat(np.expand_dims(delta_z_rho, 1), len(transect_ds.distance), axis=1)
    delta_z_rho_2d[l_below_seafloor] = np.nan
    transect_ds['delta_z'] = (['z', 'distance'], delta_z_rho_2d)
    
    depth_mean_density = np.nansum(transect_ds.density.values * transect_ds.delta_z.values, axis=0) / np.nansum(transect_ds.delta_z.values, axis=0)
    transect_ds['depth_mean_density'] = (['distance'], depth_mean_density)
    
    # -- add vertical density difference
    drho_z = np.diff(transect_ds.density.values, axis=0)
    # convert vertical density difference back to rho-points
    drho_z_rho = np.empty(transect_ds.density.shape) * np.nan
    drho_z_rho[1:-1, :] = 0.5 * (drho_z[0:-1, :] + drho_z[1:, :])
    drho_z_rho[0, :] = drho_z[0, :]
    drho_z_rho[-1, :] = drho_z[-1, :]
    transect_ds['drho_z'] = (['z', 'distance'], drho_z_rho)
    
    # --- add horizontal density difference
    drho_x = np.diff(transect_ds.density.values, axis=1)
    drho_dx = drho_x / dx
    # convert drho_dx back to rho-points
    drho_dx_rho = np.empty(transect_ds.density.shape) * np.nan
    drho_dx_rho[:, 1:-1] = 0.5 * (drho_dx[:, 0:-1] + drho_dx[:, 1:])
    drho_dx_rho[:, 0] = drho_dx[:, 0]
    drho_dx_rho[:, -1] = drho_dx[:, -1]
    transect_ds['drho_dx'] = (['z', 'distance'], drho_dx_rho)
    
    # --- add depth mean horizontal density gradient
    drho_zmean = np.diff(transect_ds.depth_mean_density.values, axis=0)
    drho_dx_zmean = drho_zmean / dx
    # convert drho_dx_zmean back to rho-points
    drho_dx_zmean_rho = np.empty(transect_ds.distance.shape) * np.nan
    drho_dx_zmean_rho[1:-1] = 0.5 * (drho_dx_zmean[0:-1] + drho_dx_zmean[1:])
    drho_dx_zmean_rho[0] = drho_dx_zmean[0]
    drho_dx_zmean_rho[-1] = drho_dx_zmean[-1]
    transect_ds['drho_dx_zmean'] = (['distance'], drho_dx_zmean_rho)
    
    return transect_ds    
    