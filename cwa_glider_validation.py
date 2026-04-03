from readers.read_ocean_data import load_roms_data, select_roms_transect_from_known_coordinates
from tools.roms import get_eta_xi_of_lon_lat_point
from readers.read_glider_data import GliderData, get_glider_transect_data
from dswt.dswt_detection import determine_dswt_along_transect

from tools.config import read_config, Config
from tools.files import get_dir_from_json, get_daily_files_in_time_range, create_dir_if_does_not_exist
from tools.coordinates import get_unique_coordinates
from tools.timeseries import get_l_time_range
from plot_tools.general import add_subtitle, get_vmin_vmax

from datetime import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from plot_tools.basic_maps import plot_basic_map
import cartopy.crs as ccrs
import cmocean as cm
import pandas as pd

def determine_dswt_along_glider_transect(transect_ds:xr.Dataset, config:Config):
    '''Conditions:
    1. Mean drho/dx < 0
    2. drho * s / rho0 < -2 10**-8
    3. transport in the bottom layer must be offshore (if available in glider data)'''
    
    RHO0 = 1025.
    
    # remove data for depths above filter_depth_up_to
    if config.filter_depth is not None:
        # replace with NaNs all values where depth > filter_depth:
        l_depth = abs(transect_ds.h.values) < config.filter_depth
        # transect_ds.slope.values[~l_depth] = np.nan
        transect_ds.drho_z.values[:, ~l_depth] = np.nan
        transect_ds.drho_dx_zmean.values[~l_depth] = np.nan
    
    # condition 1: mean depth mean horizontal density gradient (away from coast) must be negative
    mean_drhodx = np.nanmean(transect_ds.drho_dx_zmean.values)
    drhodx_condition = mean_drhodx < 0. # drhodx_condition: [ocean_time]
    
    if drhodx_condition == True:
        # condition 2: vertical density gradient needs to be sufficiently large
        # drho_s = (transect_ds.drho_z.values * transect_ds.slope.values) / RHO0
        # drho_s_condition = drho_s < -2*10**-8 # drho_s_condition: [ocean_time, s_rho, distance]
        drho_s = (transect_ds.drho_z.values / transect_ds.delta_z.values) / RHO0
        drho_s_condition = drho_s < -5*10**-5 # drho_s_condition: [ocean_time, s_rho, distance]
        # consider only vertical density gradient in bottom layers (remove any True values from surface layers)
        minimum_depth = transect_ds.h.values + config.drhodz_depth_percentage * abs(transect_ds.h.values) # note: h is negative
        z = np.repeat(np.expand_dims(transect_ds.z.values, 1), len(transect_ds.distance), axis=1)
        l_shallow = z > minimum_depth
        drho_s_condition[l_shallow] = False
        
        l_dswt = drho_s_condition
        
        thickness_dswt = np.zeros(len(transect_ds.distance))
        min_drho_s = np.zeros(len(transect_ds.distance))
        mean_drho_s = np.zeros(len(transect_ds.distance))
        
        x_dswt = np.where(np.any(l_dswt, axis=0))[0]
        for x in x_dswt:
            z_dswt = np.where(l_dswt[:, x] == True)[0][-1] # shallowest layer up to which DSWT extends
            thickness_dswt[x] = np.nansum(transect_ds.delta_z.values[0:z_dswt + 1, x])
            min_drho_s[x] = np.nanmin(drho_s[0:z_dswt + 1, x])
            mean_drho_s[x] = np.nanmean(drho_s[0:z_dswt + 1, x])
        
        return (transect_ds.distance.values[x_dswt], transect_ds.lon.values[x_dswt], transect_ds.lat.values[x_dswt],
                abs(transect_ds.h.values[x_dswt]), thickness_dswt[x_dswt], mean_drhodx, mean_drho_s[x_dswt], min_drho_s[x_dswt])
        
    return (np.array([np.nan]), np.array([np.nan]),
            np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), mean_drhodx,
            np.array([np.nan]), np.array([np.nan]))

def plot_glider_model_dswt(glider_ds:xr.Dataset, model_ds:xr.Dataset, config:Config,
                           output_path=None, show=False,
                           vmin1=None, vmax1=None, vmin2=None, vmax2=None, vmin3=None, vmax3=None):

    time_str = f'{pd.to_datetime(glider_ds.time.values[0]).strftime("%d-%m-%Y %H:%M")} - {pd.to_datetime(glider_ds.time.values[-1]).strftime("%d-%m-%Y %H:%M")}'

    # --- Glider transect
    (glider_dswt_x, _, _, glider_dswt_h, glider_dswt_dz, _, _, _) = determine_dswt_along_glider_transect(glider_ds, config)

    glider_distance = glider_ds.distance.values
    glider_z = glider_ds.z.values
    
    z_bottom = glider_ds.h.values
    density = glider_ds.density.values
    temp = glider_ds.temp.values
    salt = glider_ds.salt.values
    
    depth_ticks = np.arange(-200, 20, 20)
    ylim = [np.nanmin(glider_ds.z.values), 0]
    
    fig = plt.figure(figsize=(10, 5))
    
    def _plot_glider_transect(ax, values, vmin, vmax, cmap):
        c = ax.pcolormesh(glider_distance, glider_z, values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.plot(glider_distance, z_bottom, '-k')
        ax.fill_between(glider_distance, glider_z[0], z_bottom, color='#d2d2d2')
        ax.set_ylabel('Depth (m)')
        ax.set_yticks(depth_ticks)
        ax.set_xlim([0, glider_distance[-1]])
        ax.set_ylim(ylim)
        ax.set_xticklabels([])
        ax.set_xlabel('')
        return c
    
    # density
    if np.logical_or(vmin1==None, vmax1==None):
        vmin1, vmax1 = get_vmin_vmax(density-1000, min_bin=24, max_bin=27, dbin=0.05)
    ax1 = plt.subplot(2, 3, 1)
    c1 = _plot_glider_transect(ax1, density-1000, vmin1, vmax1, cm.cm.thermal_r)
    ax1.scatter(glider_dswt_x, -glider_dswt_h + glider_dswt_dz, marker='o', c='w', s=40)
    ax1.scatter(glider_dswt_x, -glider_dswt_h + glider_dswt_dz, marker='o', c='k', s=20)
    add_subtitle(ax1, f'(a) Ocean glider', location='lower left')
    
    # temperature
    if np.logical_or(vmin2==None, vmax2==None):
        vmin2, vmax2 = get_vmin_vmax(temp, min_bin=np.nanmin(temp), max_bin=np.nanmax(temp))
    ax2 = plt.subplot(2, 3, 2)
    c2 = _plot_glider_transect(ax2, temp, vmin2, vmax2, 'RdYlBu_r')
    ax2.set_yticklabels([])
    ax2.set_ylabel('')
    add_subtitle(ax2, f'(b) Ocean glider', location='lower left')
    
    # salinity
    if np.logical_or(vmin3==None, vmax3==None):
        vmin3, vmax3 = get_vmin_vmax(salt, min_bin=np.nanmin(salt), max_bin=np.nanmax(salt), dbin=0.02)
    ax3 = plt.subplot(2, 3, 3)
    c3 = _plot_glider_transect(ax3, salt, vmin3, vmax3, cm.cm.haline)
    ax3.set_yticklabels([])
    ax3.set_ylabel('')
    add_subtitle(ax3, f'(c) Ocean glider', location='lower left')
    
    # --- Model transect
    (model_t_dswt, _, _, model_dswt_dz, _, model_dswt_x, _, _, model_dswt_h, _, _, _, _) = determine_dswt_along_transect(model_ds, config, mld_condition=False)
    
    model_x = model_ds.distance.values
    model_z = model_ds.z_rho.values
    if len(model_t_dswt) != 0:
        model_density = np.nanmean(model_ds.density.values[model_t_dswt, :, :], axis=0)
        model_temp = np.nanmean(model_ds.temp.values[model_t_dswt, :, :], axis=0)
        model_salt = np.nanmean(model_ds.salt.values[model_t_dswt, :, :], axis=0)
    else:
        model_density = np.nanmean(model_ds.density.values, axis=0)
        model_temp = np.nanmean(model_ds.temp.values, axis=0)
        model_salt = np.nanmean(model_ds.salt.values, axis=0)
        
    model_h = -model_ds.h.values
    ymin_model = np.nanmin([ylim[0], np.nanmin(model_z)])
    
    def _plot_model_transect(ax, values, vmin, vmax, cmap):
        c = ax.pcolormesh(model_x, model_z, values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.fill_between(model_x, ymin_model, model_h, color='#d2d2d2', edgecolor='k')
        ax.set_ylabel('Depth (m)')
        ax.set_yticks(depth_ticks)
        ax.set_xlim([0, glider_distance[-1]])
        ax.set_ylim([ymin_model, 0])
        ax.set_xlabel('Distance along transect (m)')
        return c
    
    # density
    ax4 = plt.subplot(2, 3, 4)
    _plot_model_transect(ax4, model_density-1000, vmin1, vmax1, cm.cm.thermal_r)
    ax4.scatter(model_dswt_x, -model_dswt_h + model_dswt_dz, marker='o', c='w', s=40)
    ax4.scatter(model_dswt_x, -model_dswt_h + model_dswt_dz, marker='o', c='k', s=20)
    add_subtitle(ax4, '(d) CWA-ROMS', location='lower left')
    
    # temperature
    ax5 = plt.subplot(2, 3, 5)
    _plot_model_transect(ax5, model_temp, vmin2, vmax2, 'RdYlBu_r')
    ax5.set_yticklabels([])
    ax5.set_ylabel('')
    add_subtitle(ax5, '(e) CWA-ROMS', location='lower left')
    
    # salinity
    ax6 = plt.subplot(2, 3, 6)
    _plot_model_transect(ax6, model_salt, vmin3, vmax3, cm.cm.haline)
    ax6.set_yticklabels([])
    ax6.set_ylabel('')
    add_subtitle(ax6, '(f) CWA-ROMS', location='lower left')
    
    # --- Colorbars
    def _plot_colorbar(ax, c, label):
        ll, bb, ww, hh = ax.get_position().bounds
        cax = fig.add_axes([ll, bb-0.16, ww, 0.04])
        cbar = plt.colorbar(c, cax=cax, orientation='horizontal')
        cbar.set_label(label)
    
    _plot_colorbar(ax4, c1, '$\sigma_T$ (kg m$^{-3}$)')
    _plot_colorbar(ax5, c2, 'Temperature ($^o$C)')
    _plot_colorbar(ax6, c3, 'Salinity')
    
    # --- Maps with transect
    l1, b1, w1, h1 = ax1.get_position().bounds
    axm1 = fig.add_axes([l1+0.05*w1, b1+0.2*h1, w1/5, h1/3], projection=ccrs.PlateCarree())
    plot_basic_map(axm1, [114.8, 116.0], [-32.5, -31.0])
    axm1.plot(glider_ds.lon.values, glider_ds.lat.values, '-', color='#C70039', linewidth=1)
    axm1.set_xticks([])
    axm1.set_yticks([])
    
    l4, b4, w4, h4 = ax4.get_position().bounds
    axm2 = fig.add_axes([l4+0.05*w4, b4+0.2*h4, w4/5, h4/3], projection=ccrs.PlateCarree())
    plot_basic_map(axm2, [114.8, 116.0], [-32.5, -31.0])
    axm2.plot(model_ds.lon_rho.values, model_ds.lat_rho.values, '-', color='#C70039', linewidth=1)
    axm2.set_xticks([])
    axm2.set_yticks([])
    
    # title
    plt.suptitle(time_str, x=0.5, y=0.92, ha='center')
    
    if output_path is not None:
        # plt.savefig(output_path, bbox_extra_artists=(qkey,), bbox_inches='tight', dpi=300)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()
    
if __name__ == '__main__':
    glider_dir = get_dir_from_json('glider_data')
    
    input_files = [# may 2010
                   "IMOS_ANFOG_BCEOPSTUV_20100507T030402Z_SL130_FV01_timeseries_END-20100515T133815Z.nc",
                   # july 2010
                   "IMOS_ANFOG_BCEOPSTUV_20100628T045246Z_SL130_FV01_timeseries_END-20100714T233721Z.nc"]
                #    # sep 2012
                #    "IMOS_ANFOG_BCEOPSTUV_20120824T031255Z_SL248_FV01_timeseries_END-20120914T035136Z.nc",
                #    # may 2016
                #    "IMOS_ANFOG_BCEOPSTUV_20160512T034541Z_SL502_FV01_timeseries_END-20160530T020243Z.nc",
                #    # jul 2019
                #    "IMOS_ANFOG_BCEOPSTUV_20190625T071857Z_SL248_FV01_timeseries_END-20190723T040704Z.nc",
                #    # jul 2020
                #    "IMOS_ANFOG_BCEOPSTUV_20200625T074407Z_SL248_FV01_timeseries_END-20200721T053759Z.nc",
                #    # jul 2022
                #    "IMOS_ANFOG_BCEOPSTUV_20220628T064224Z_SL286_FV01_timeseries_END-20220712T082641Z.nc"]
    
    start_dates = [datetime(2010, 5, 12, 10, 0),
                   datetime(2010, 7, 5, 14, 0)]
                #    datetime(2012, 9, 10, 0, 0),
                #    datetime(2016, 5, 13, 8, 0),
                #    datetime(2019, 7, 3, 6, 0),
                #    datetime(2020, 7, 5),
                #    datetime(2022, 7, 5, 0, 0)]
    end_dates = [datetime(2010, 5, 14, 0, 0),
                 datetime(2010, 7, 8, 0, 0)]
                #  datetime(2012, 9, 11, 22, 0),
                #  datetime(2016, 5, 15, 0, 0),
                #  datetime(2019, 7, 6, 0, 0),
                #  datetime(2020, 7, 6, 10, 0),
                #  datetime(2022, 7, 6, 5, 0)]
    flip = [True, False]#, True, False, False, False, False, False]
    vmin1 = [24.8, 24.8]
    vmax1 = [25.6, 25.6]
    vmin2 = [20.0, 19.0]
    vmax2 = [22.0, 21.0]
    vmin3 = [35.4, 35.4]
    vmax3 = [35.7, 35.6]
    
    config = read_config('cwa')
    
    model_dir = get_dir_from_json("cwa")
    model_grid_file = f'{model_dir}grid.nc'
    
    output_dir = f'{get_dir_from_json("plots")}cwa/'
    create_dir_if_does_not_exist(output_dir)
    
    for i, input_file in enumerate(input_files):
        glider_data = GliderData.read_from_netcdf(f'{glider_dir}{input_file}')
        glider_data.get_data_in_time_frame(start_dates[i], end_dates[i])
        glider_ds = get_glider_transect_data(glider_data, flip=flip[i])
        
        model_files = get_daily_files_in_time_range(f'{model_dir}{start_dates[i].year}/', start_dates[i], end_dates[i], 'nc')
        roms_ds = load_roms_data(model_files, model_grid_file)
        etas, xis = get_eta_xi_of_lon_lat_point(roms_ds.lon_rho.values, roms_ds.lat_rho.values, glider_ds.lon.values, glider_ds.lat.values)
        eta, xi = get_unique_coordinates(etas, xis)
        if flip[i] == True:
            eta = eta[::-1]
            xi = xi[::-1]
        model_ds = select_roms_transect_from_known_coordinates(roms_ds, eta, xi)
        model_time = np.array([pd.to_datetime(t) for t in model_ds.ocean_time.values])
        l_time = get_l_time_range(model_time, start_dates[i], end_dates[i])
        model_ds = model_ds.isel(ocean_time=np.where(l_time)[0])
        
        output_path = f'{output_dir}glider_{pd.to_datetime(start_dates[i]).strftime("%Y%m%d")}.jpg'
        plot_glider_model_dswt(glider_ds, model_ds, config, output_path=output_path,
                               vmin1=vmin1[i], vmax1=vmax1[i], vmin2=vmin2[i], vmax2=vmax2[i], vmin3=vmin3[i], vmax3=vmax3[i])
        