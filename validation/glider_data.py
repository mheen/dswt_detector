import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.files import get_dir_from_json, get_files_in_dir
from tools.config import read_config, Config
from tools.coordinates import get_distance_between_points, get_bearing_between_points
from tools.seawater_density import calculate_density
from tools.timeseries import convert_datetime_to_time, get_l_time_range
from tools.peak_detect import peak_detect
from tools.roms import get_distance_along_transect
from readers.read_ocean_data import calculate_down_transect_velocity_component
from plot_tools.basic_maps import plot_basic_map
from tools import log
from gridfit import gridfit
from scipy import interpolate
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

USE_QC_FLAGS = [1, 2]

lon_range_default = [114.5, 116.]
lat_range_default = [-33., -31.]
meridians_default = [115., 116.]
parallels_default = [-33., -32., -31.]

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
                 u:np.ndarray,
                 v:np.ndarray):

        self.time = time
        self.lon = lon
        self.lat = lat
        self.depth = depth
        self.temp = temp
        self.salt = salt
        self.ox2 = ox2
        self.cphl = cphl
        self.bbp = bbp
        self.u = u
        self.v = v

        self.add_density()
        self.add_cumulative_time_along_glider_path()
        self.add_bottom(show=False)
        
    def add_density(self):
        self.density = calculate_density(self.salt, self.temp, self.depth)

    def add_cumulative_time_along_glider_path(self):
        self.cumtime, _ = convert_datetime_to_time(self.time, time_units='days', time_origin=self.time[0])

    def add_bottom(self, show=True) -> np.ndarray:
        '''Determines approximate ocean bottom by finding the bottom of each glider dive.'''
        max_tab, _ = peak_detect(self.depth, 15) # 15 m as estimate of how big the peaks are to look for
        i_max = max_tab[:, 0].astype(int)
        max_values = max_tab[:, 1]

        # throw out points that are shallow dives
        diff_max = np.diff(max_values)
        i_shallow_dives = np.where(diff_max>=10)[0]
        i_max = np.delete(i_max, i_shallow_dives)
        max_values = np.delete(max_values, i_shallow_dives)

        # interpolate to get bottom values
        f = interpolate.PchipInterpolator(self.cumtime[i_max], -self.depth[i_max])
        self.z_bottom = f(self.cumtime)

        if show is True:
            ax = plt.axes()
            ax.plot(self.cumtime[i_max], -self.depth[i_max], 'xk')
            ax.plot(self.cumtime, self.z_bottom, '-k')
            plt.show()

    def get_transect_data(self, values, dt=1/24, dz=1):
        '''Interpolates glider data to a full cross section.'''
        # create grid along transect to interpolate to
        t = np.arange(np.nanmin(self.cumtime), np.nanmax(self.cumtime), dt)
        z = np.arange(np.nanmin(-self.depth), 0, dz)
        
        if all(np.isnan(values)):
            return t, z, np.empty((len(z), len(t))) * np.nan
        
        values_fitted, _, _ = gridfit(self.cumtime, -self.depth, values, t, z)

        return t, z, values_fitted

    def get_data_in_time_frame(self, start_time:datetime, end_time:datetime):
        l_time = get_l_time_range(self.time, start_time, end_time)
        return GliderData(self.time[l_time], self.lon[l_time], self.lat[l_time],
                          self.depth[l_time], self.temp[l_time], self.salt[l_time],
                          self.ox2[l_time], self.cphl[l_time], self.bbp[l_time],
                          self.u[l_time], self.v[l_time])

    def plot_transect(self, parameter='density',
                      cmap='RdBu_r', vmin=None, vmax=None,
                      dz_interp=1, dt_interp=1/24, fill_color='#989898'):
        '''Plots full transect based on fitted glider'''
        if parameter.lower().startswith('t'):
            values = self.temp
            cbar_label = 'Temperature (C)'
        elif parameter.lower().startswith('s'):
            values = self.salt
            cbar_label = 'Salinity (ppt)'
        elif parameter.lower().startswith('d'):
            values = self.density-1000
            cbar_label = 'sigma_T'
        elif parameter.lower().startswith('o'):
            values = self.ox2
            cbar_label = 'Oxygen (umol/kg)'
        elif parameter.lower().startswith('c'):
            values = self.cphl
            cbar_label = 'Chlorophyll (mg/m$^3$)'
        elif parameter.lower().startswith('b'):
            values = self.bbp
            cbar_label = 'Particle backscatter (m$^{-1}$)'
        elif parameter.lower().startswith('v'):
            values = np.sqrt(self.u**2+self.v**2)
            cbar_label = 'Velocity (m/s)'
        else:
            raise ValueError(f'Unknown parameter requested for transect: {parameter}')

        t, z, transect_values = self.get_transect_data(values, dt=dt_interp, dz=dz_interp)

        fig = plt.figure(figsize=(8, 3))
        ax = plt.axes()

        tt, zz = np.meshgrid(t, z)
        
        # get overall distance of transect
        l_nonan = np.logical_and(~np.isnan(self.lon), ~np.isnan(self.lat))
        lon = self.lon[l_nonan]
        lat = self.lat[l_nonan]
        max_dist = get_distance_between_points(lon[0], lat[0], lon[-1], lat[-1])
        
        if all(self.lon[~np.isnan(self.lon)] < 130.): # in WA: so flip transect horizontally to show coast on the east
            transect_values = np.fliplr(transect_values)
            z_bottom = np.flip(np.copy(self.z_bottom))
            x_label = 'Distance along transect (km)'
            xticklabels = [np.round(max_dist/1000, 0), 0]
        else:
            z_bottom = self.z_bottom
            x_label = 'Distance along transect (km)'
            xticklabels = [0, np.round(max_dist/1000, 0)]

        c = ax.pcolormesh(tt, zz, transect_values, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(c)
        cbar.set_label(cbar_label)
        ax.plot(self.cumtime, z_bottom, '-k')
        ax.fill_between(self.cumtime, z[0], z_bottom, color=fill_color)
        
        ax.set_xlim([0, self.cumtime[-1]])
        ax.set_ylim([z[0], 0])

        ax.set_ylabel('Depth (m)')
        ax.set_xlabel(x_label)
        
        ax.set_xticks([0, self.cumtime[-1]])
        ax.set_xticklabels(xticklabels)

        plt.show()

    def plot_track(self, show_labels=True, color='k'):
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, lon_range=lon_range_default, lat_range=lat_range_default,
                            meridians=meridians_default, parallels=parallels_default)
        
        l_nonans_position = np.logical_and(~np.isnan(self.lon), ~np.isnan(self.lat))
        ax.plot(self.lon[l_nonans_position][0], self.lat[l_nonans_position][0],
                'or', markersize=5, label='Initial location',
                transform=ccrs.PlateCarree(), zorder=5)
        ax.plot(self.lon[l_nonans_position][-1], self.lat[l_nonans_position][-1],
                'xr', markersize=5, label='Final location',
                transform=ccrs.PlateCarree(), zorder=6)
        ax.plot(self.lon, self.lat, '.', color=color, transform=ccrs.PlateCarree())
        
        if show_labels is True:
            # label locations with date
            time_dates = np.array([t.date() for t in self.time])
            n_days = (time_dates[-1]-time_dates[0]).days
            i_label = []
            for i in range(n_days+1):
                i_date = np.where(np.logical_and(time_dates==time_dates[0]+timedelta(days=i), l_nonans_position))[0]
                if any(i_date):
                    i_label.append(i_date[0])
            time_labels = [t.strftime('%d-%m-%Y %H:%M') for t in self.time[i_label]]
            for i in range(len(i_label)):
                ax.text(self.lon[i_label[i]], self.lat[i_label[i]], time_labels[i], transform=ccrs.PlateCarree())

        ax.legend(loc='upper right')
        plt.show()
    
    @staticmethod
    def read_from_netcdf(input_path:str):
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
        # l_time = sum([ds.TIME_quality_control.values == qc_flag for qc_flag in USE_QC_FLAGS]).astype(bool)
        # time[~l_time] = np.nan
        
        lon = ds.LONGITUDE.values
        l_lon = sum([ds.LONGITUDE_quality_control.values == qc_flag for qc_flag in USE_QC_FLAGS]).astype(bool)
        lon[~l_lon] = np.nan
        
        lat = ds.LATITUDE.values
        l_lat = sum([ds.LATITUDE_quality_control.values == qc_flag for qc_flag in USE_QC_FLAGS]).astype(bool)
        lat[~l_lat] = np.nan
        
        depth = ds.DEPTH.values
        l_depth = sum([ds.DEPTH_quality_control.values == qc_flag for qc_flag in USE_QC_FLAGS]).astype(bool)
        depth[~l_depth] = np.nan

        temp = ds.TEMP.values
        l_temp = sum([ds.TEMP_quality_control.values == qc_flag for qc_flag in USE_QC_FLAGS]).astype(bool)
        temp[~l_temp] = np.nan

        salt = ds.PSAL.values
        l_salt = sum([ds.PSAL_quality_control.values == qc_flag for qc_flag in USE_QC_FLAGS]).astype(bool)
        salt[~l_salt] = np.nan

        if 'DOX2' in list(ds.variables.keys()):
            ox2 = ds.DOX2.values
            l_ox2 = sum([ds.DOX2_quality_control.values == qc_flag for qc_flag in USE_QC_FLAGS]).astype(bool)
            ox2[~l_ox2] = np.nan
        else:
            ox2 = np.empty(time.shape) * np.nan

        if 'CPHL' in list(ds.variables.keys()):
            cphl = ds.CPHL.values
            l_cphl = sum([ds.CPHL_quality_control.values == qc_flag for qc_flag in USE_QC_FLAGS]).astype(bool)
            cphl[~l_cphl] = np.nan
        else:
            cphl = np.empty(time.shape) * np.nan

        if 'BBP' in list(ds.variables.keys()):
            bbp = ds.BBP.values
            l_bbp = sum([ds.BBP_quality_control.values == qc_flag for qc_flag in USE_QC_FLAGS]).astype(bool)
            bbp[~l_bbp] = np.nan
        else:
            bbp = np.empty(time.shape) * np.nan
        
        if 'UCUR' in list(ds.variables.keys()):
            u = ds.UCUR.values
            l_u = sum([ds.UCUR_quality_control.values == qc_flag for qc_flag in USE_QC_FLAGS]).astype(bool)
            u[~l_u] = np.nan
        else:
            u = np.empty(time.shape) * np.nan

        if 'VCUR' in list(ds.variables.keys()):
            v = ds.VCUR.values
            l_v = sum([ds.VCUR_quality_control.values == qc_flag for qc_flag in USE_QC_FLAGS]).astype(bool)
            v[~l_v] = np.nan
        else:
            v = np.empty(time.shape) * np.nan
        
        return GliderData(time, lon, lat, depth, temp, salt, ox2, cphl, bbp, u, v)

def extract_transect_data_and_write_to_netcdf(transect_data:GliderData, output_path:str):
    time, z, density = transect_data.get_transect_data(transect_data.density)
    _, _, temp = transect_data.get_transect_data(transect_data.temp)
    _, _, salt = transect_data.get_transect_data(transect_data.salt)
    _, _, u = transect_data.get_transect_data(transect_data.u)
    _, _, v = transect_data.get_transect_data(transect_data.v)
    
    l_loc = np.logical_and(~np.isnan(transect_data.lon), ~np.isnan(transect_data.lat))
    l_time = np.logical_and(time >= np.nanmin(transect_data.cumtime[l_loc]), time <= np.nanmax(transect_data.cumtime[l_loc]))
    
    f_lon = interpolate.interp1d(transect_data.cumtime[l_loc], transect_data.lon[l_loc])
    lon = f_lon(time[l_time])
    f_lat = interpolate.interp1d(transect_data.cumtime[l_loc], transect_data.lat[l_loc])
    lat = f_lat(time[l_time])
    f_h = interpolate.interp1d(transect_data.cumtime[~np.isnan(transect_data.z_bottom)], transect_data.z_bottom[~np.isnan(transect_data.z_bottom)])
    h = f_h(time[l_time])
    
    distance = get_distance_along_transect(lon, lat)
    
    dates = np.array([transect_data.time[0] + timedelta(days=t) - timedelta(hours=8) for t in time[l_time]]) # -8 hours to convert to UTC
    
    ds = xr.Dataset(
        data_vars={
            "temp": (("z", "distance"), temp[:, l_time]),
            "salt": (("z", "distance"), salt[:, l_time]),
            "density": (("z", "distance"), density[:, l_time]),
            "u": (("z", "distance"), u[:, l_time]),
            "v": (("z", "distance"), v[:, l_time]),
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
    
    ds.to_netcdf(output_path)

if __name__ == '__main__':
    input_dir = get_dir_from_json('glider_data_org')
    output_dir = get_dir_from_json('glider_transects')
    
    with open('validation/input/glider_transects.json', 'r') as f:
        transect_info = json.load(f)
    ncfiles = list(transect_info.keys())
    
    for ncfile in ncfiles:
        glider_data = GliderData.read_from_netcdf(f'{input_dir}{ncfile}')
        start_dates = [datetime.strptime(t, '%d/%m/%Y %H:%M') for t in transect_info[ncfile]['start']]
        end_dates = [datetime.strptime(t, '%d/%m/%Y %H:%M') for t in transect_info[ncfile]['end']]
        
        for t in range(len(start_dates)):
            filename = f'{start_dates[t].strftime("%Y%m%d")}.nc'
            output_path = f'{output_dir}{filename}'
            if os.path.exists(output_path):
                continue
            
            transect_data = glider_data.get_data_in_time_frame(start_dates[t], end_dates[t])
            
            log.info(f'Writing transect data to file: {output_path}')
            extract_transect_data_and_write_to_netcdf(transect_data, output_path)
    