from transects import read_transects_in_lon_lat_range_from_json
from readers.read_ocean_data import load_roms_data, select_roms_transect_from_known_coordinates, select_input_files
from tools.roms import get_eta_xi_of_lon_lat_point
from tools import log
from readers.read_glider_data import GliderData, get_glider_transect_data
from readers.read_dswt_output import read_df_from_multiple_csvs
from dswt.dswt_detection import determine_dswt_along_transect

from tools.config import read_config, Config
from tools.files import get_dir_from_json, get_daily_files_in_time_range, create_dir_if_does_not_exist
from tools.coordinates import get_unique_coordinates
from tools.timeseries import get_l_time_range, get_closest_time_index, get_closest_time_index_array
from plot_tools.general import add_subtitle, get_vmin_vmax

from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from plot_tools.basic_maps import plot_basic_map
import cartopy.crs as ccrs
import cmocean as cm
import pandas as pd
import os

color_dswt = '#25419e'
color_dswt_std = "#6573a1"
color_transport = '#0e6e22'
color_pos = '#900C3F'

glider_dir = get_dir_from_json('glider_data')
model_dir = get_dir_from_json("cwa")
model_grid_file = f'{model_dir}grid.nc'

config = read_config('cwa')

# ----------------------------------------------------------------------------------
# Functions for vertical profile during DSWT comparison
# ----------------------------------------------------------------------------------
def get_glider_obs_of_dswt_vertical_profile(glider_data:GliderData, glider_dswt_t):
    # interpolate lon and lat values when glider not at surface
    l_nans_org = np.logical_or(np.logical_or(np.isnan(glider_data.depth), np.isnan(glider_data.lon)), np.isnan(glider_data.lat))
    time_obs = glider_data.cumtime[~l_nans_org]
    lon_obs = glider_data.lon[~l_nans_org]
    lat_obs = glider_data.lat[~l_nans_org]

    glider_lon = np.interp(glider_data.cumtime, time_obs, lon_obs)
    glider_lat = np.interp(glider_data.cumtime, time_obs, lat_obs)
    
    # get indices of DSWT dive
    t_dswt = get_closest_time_index(pd.to_datetime(glider_data.time), pd.to_datetime(glider_dswt_t))
    dive_code = glider_data.phase[t_dswt] # 1: descending profile, 4: ascending profile (0: surface drifting, 3: inflexion)
    i_dive_change = np.where((glider_data.phase - dive_code) != 0)[0] # indices with same dive code
    i_start_dive = t_dswt + np.max((i_dive_change - t_dswt)[(i_dive_change - t_dswt) < 0]) + 1
    i_stop_dive = t_dswt + np.min((i_dive_change - t_dswt)[(i_dive_change - t_dswt) > 0]) - 1
    
    # select subset of data
    time = glider_data.time[i_start_dive:i_stop_dive]
    depth = glider_data.depth[i_start_dive:i_stop_dive]
    lon = glider_lon[i_start_dive:i_stop_dive]
    lat = glider_lat[i_start_dive:i_stop_dive]
    temp = glider_data.temp[i_start_dive:i_stop_dive]
    salt = glider_data.salt[i_start_dive:i_stop_dive]
    density = glider_data.density[i_start_dive:i_stop_dive]    
    
    return time, depth, lon, lat, temp, salt, density

def _nearest_sigma_index(z_rho, i_time, eta, xi, glider_depth):
    glider_z = -np.abs(glider_depth)

    profiles = z_rho[:, eta, xi].T

    dz = np.abs(profiles - glider_z[:, None])
    i_s = np.argmin(dz, axis=1)
    return i_s

def match_glider_obs_to_model(glider_time, glider_lon, glider_lat, glider_depth, glider_temp, glider_salt, glider_density,
                              model_ds:xr.Dataset, output_path=None, append=True):
    l_nans = np.logical_or(np.isnan(glider_lon), np.isnan(glider_lat))

    etas, xis = get_eta_xi_of_lon_lat_point(model_ds.lon_rho.values, model_ds.lat_rho.values, glider_lon[~l_nans], glider_lat[~l_nans])
    i_times = get_closest_time_index_array(pd.to_datetime(model_ds.ocean_time.values), pd.to_datetime(glider_time[~l_nans]))
    i_sigma = _nearest_sigma_index(model_ds.z_rho.values, i_times, etas, xis, glider_depth[~l_nans])

    glider_vars = ['temp', 'salt', 'density']

    df = pd.DataFrame()
    df['eta'] = etas
    df['xi'] = xis
    df['i_time'] = i_times
    df['i_s'] = i_sigma
    df['temp'] = glider_temp[~l_nans]
    df['salt'] = glider_salt[~l_nans]
    df['density'] = glider_density[~l_nans]

    group_cols = ['i_time', 'eta', 'xi', 'i_s']
    agg_dict = {v: ['mean', 'std', 'count'] for v in glider_vars}
    binned = df.groupby(group_cols).agg(agg_dict)
    binned.columns = ['_'.join(col).strip() for col in binned.columns.values]
    binned = binned.reset_index()

    binned['lon'] = np.empty(len(binned)) * np.nan
    binned['lat'] = np.empty(len(binned)) * np.nan
    binned['ocean_time'] = np.empty(len(binned)) * np.nan
    binned['z'] = np.empty(len(binned)) * np.nan

    for v in glider_vars:
        binned[f'model_{v}'] = np.empty(len(binned)) * np.nan

    for i in range(len(binned)):
        binned.loc[i, 'ocean_time'] = model_ds['ocean_time'][binned['i_time'][i]].values
        binned.loc[i, 'lon'] = model_ds['lon_rho'][binned['eta'][i], binned['xi'][i]].values
        binned.loc[i, 'lat'] = model_ds['lat_rho'][binned['eta'][i], binned['xi'][i]].values
        binned.loc[i, 'z'] = model_ds['z_rho'][binned['i_s'][i], binned['eta'][i], binned['xi'][i]].values

        for v in glider_vars:
            binned.loc[i, f'model_{v}'] = model_ds[v][binned['i_time'][i], binned['i_s'][i], binned['eta'][i], binned['xi'][i]]

    if output_path is not None:
        log.info(f'Writing glider and model data to csv: {output_path}')
        if os.path.exists(output_path):
            binned.to_csv(output_path, index=False, mode='a', header=False)
        else:
            binned.to_csv(output_path, index=False)

    return binned

def compare_glider_model_vertical_profiles(input_files, start_dates, end_dates):
    for i, input_file in enumerate(input_files):
        glider_data = GliderData.read_from_netcdf(f'{glider_dir}{input_file}')
        glider_data.get_data_in_time_frame(start_dates[i], end_dates[i])
        
        model_files = get_daily_files_in_time_range(f'{model_dir}{start_dates[i].year}/', start_dates[i], end_dates[i], 'nc')
        roms_ds = load_roms_data(model_files, model_grid_file)
        
        # interpolate lon and lat values when glider not at surface
        l_nans_org = np.logical_or(np.logical_or(np.isnan(glider_data.depth), np.isnan(glider_data.lon)), np.isnan(glider_data.lat))
        time_obs = glider_data.cumtime[~l_nans_org]
        lon_obs = glider_data.lon[~l_nans_org]
        lat_obs = glider_data.lat[~l_nans_org]
    
        glider_lon = np.interp(glider_data.cumtime, time_obs, lon_obs)
        glider_lat = np.interp(glider_data.cumtime, time_obs, lat_obs)
        
        # Create DataFrame with glider data
        l_nonan = ~np.isnan(glider_data.depth)
        df = pd.DataFrame({
            'cumtime': glider_data.cumtime[l_nonan],
            'time': glider_data.time[l_nonan],
            'depth': glider_data.depth[l_nonan],
            'phase': glider_data.phase[l_nonan],  # 0, 1, 4, 3 as you defined
            'lat': glider_lat[l_nonan],
            'lon': glider_lon[l_nonan],
            'temp': glider_data.temp[l_nonan],
            'salt': glider_data.salt[l_nonan],
            'density': glider_data.density[l_nonan],
        }).sort_values('cumtime').reset_index(drop=True)
        
        # Identify start of each new dive (transition to phase 1)
        df['new_dive'] = (df['phase'] == 1) & (df['phase'].shift(1) != 1)
        df['dive_id'] = df['new_dive'].cumsum()
        
        # remove dives that go to depths beyond 60 m
        for id in np.unique(df['dive_id']):
            max_dive_depth = np.nanmax(df.loc[df['dive_id'] == id]['depth'])
            if max_dive_depth > 60:
                df = df.drop(df[df['dive_id'] == id].index)
                
        # compare glider data to roms data    
        glider_roms_df = match_glider_obs_to_model(df.time.values, df.lon.values, df.lat.values,
                                                    df.depth.values, df.temp.values, df.salt.values,
                                                    df.density.values, roms_ds,
                                                    output_path='plots/cwa/cwa_glider_roms_comparison.csv')

# ----------------------------------------------------------------------------------
# Functions for drho/dx comparison
# ----------------------------------------------------------------------------------
def get_approximate_two_rocks_coordinates(glider_input_files, start_dates, end_dates):
    two_rocks_lon_min = np.empty(len(glider_input_files)) * np.nan
    two_rocks_lon_max = np.empty(len(glider_input_files)) * np.nan
    two_rocks_lat_min = np.empty(len(glider_input_files)) * np.nan
    two_rocks_lat_max = np.empty(len(glider_input_files)) * np.nan
    for i, glider_input_file in enumerate(glider_input_files):
        glider_data = GliderData.read_from_netcdf(f'{glider_dir}{glider_input_file}')
        glider_data.get_data_in_time_frame(start_dates[i], end_dates[i])
        glider_ds = get_glider_transect_data(glider_data, flip=flip[i])
        two_rocks_lon_min[i] = np.nanmin(glider_ds.lon)
        two_rocks_lon_max[i] = np.nanmax(glider_ds.lon)
        two_rocks_lat_min[i] = np.nanmin(glider_ds.lat)
        two_rocks_lat_max[i] = np.nanmax(glider_ds.lat)
    return [np.nanmin(two_rocks_lon_min), np.nanmax(two_rocks_lon_max)], [np.nanmin(two_rocks_lat_min), np.nanmax(two_rocks_lat_max)]

def calculate_drhodx_model(transects:dict, years:list, config:Config):
    transect_names = list(transects.keys())
    
    grid_file = f'{get_dir_from_json("cwa")}grid.nc'
    
    for year in years:
        log.info(f'Calculating Two Rocks drho/dx for {year}')
        input_dir = f'{get_dir_from_json("cwa")}{year}/'
        output_drhodx = f'{get_dir_from_json("analysis")}drhodx_{year}.csv'
        
        if os.path.exists(output_drhodx):
            df_temp = pd.read_csv(output_drhodx)
            time = df_temp['time'].values
            time_last = datetime.strptime(pd.unique(time)[-1], '%Y-%m-%d')
            if time_last == datetime(year, 12, 31):
                log.info(f'Output already exists for {year}, skipping.')
                continue
            else:
                log.info(f'''Output partially exists for {year}. Running from {time_last+timedelta(days=1)} onwards.
                        Please check to make sure that all transects for {time_last} were written to file.''')
                date_range = [time_last+timedelta(days=1), datetime(year, 12, 31)]
        else:
            date_range = [datetime(year, 1, 1), datetime(year, 12, 31)]
        
        roms_files = select_input_files(input_dir, file_preface='cwa_', date_range=date_range)
        roms_files.sort()

        for file in roms_files:
            ds_roms = load_roms_data(file, grid_file)
            ocean_time = pd.to_datetime(ds_roms.ocean_time.values[0]).date()
            
            drhodx = np.empty(len(transect_names)) * np.nan
            for i, transect_name in enumerate(transect_names):
                eta = transects[transect_name]['eta']
                xi = transects[transect_name]['xi']
                transect_ds = select_roms_transect_from_known_coordinates(ds_roms, eta, xi)
                
                l_depth = transect_ds.h.values < config.filter_depth
                transect_ds.drho_dx_zmean.values[:, ~l_depth] = np.nan
                drhodx[i] = np.nanmean(np.nanmean(transect_ds.drho_dx_zmean.values, axis=1), axis=0)
                
            df_drhodx = pd.DataFrame(data=np.array([[ocean_time, np.nanmean(drhodx)]]), columns=['time', 'drhodx_mean'])
            if os.path.exists(output_drhodx):
                df_drhodx.to_csv(output_drhodx, mode='a', header=False, index=False)
                log.info(f'Wrote drho/dx for {ocean_time.strftime("%d-%m-%Y")} to file: {output_drhodx}')
            else:
                df_drhodx.to_csv(output_drhodx, index=False)
                log.info(f'Wrote drho/dx for {ocean_time.strftime("%d-%m-%Y")} to file: {output_drhodx}')

def _mahjabin_2020_drhodx():
    mean = 10**-5 * np.array([0.32, 0.23, -1.25, -2.07, -1.97, -1.42, -2.02, -1.10, -0.99, -0.36, -0.72, np.nan])
    max = 10**-5 * np.array([0.73, 0.97, 0.97, -1.56, -0.79, -0.24, -2.02, 0.18, -0.99, 0.18, 0.12, np.nan])
    min = 10**-5 * np.array([-0.11, -0.53, -2.54, -2.35, -3.05, -2.28, -2.02, -1.79, -0.99, -1.16, -1.30, np.nan])
    months = np.arange(1, 13)
    df = pd.DataFrame(data=np.array([months, mean, min, max]).transpose(), columns=['month', 'mean', 'min', 'max'])
    return df

# ----------------------------------------------------------------------------------
# Validation plot
# ----------------------------------------------------------------------------------
def plot_glider_model_comparisons(input_files, start_dates, end_dates, output_path='plots/cwa/glider_cwa_validation.jpg', show=False):
    
    # --- Monthly mean drhodx comparison ---
    years=np.arange(2008, 2019)
    
    df = read_df_from_multiple_csvs(get_dir_from_json('analysis'), years, 'drhodx_')
    df['time'] = pd.to_datetime(df['time'])
    df_monthly = df.groupby(df['time'].dt.month).agg(
        mean=('drhodx_mean', 'mean'),
        min=('drhodx_mean', 'min'),
        max=('drhodx_mean', 'max')
    ).reset_index()
    df_monthly = df_monthly.rename(columns={'time': 'month'})
    df_mahjabin = _mahjabin_2020_drhodx()
    
    # --- Direct observations comparison ---
    glider_roms_df = pd.read_csv('plots/cwa/cwa_glider_roms_comparison.csv')
    # --- RMSE mean vertical profile
    def _rmse(df):
        rmse_density = np.sqrt(np.nanmean((df['model_density'] - df['density_mean'])**2))
        rmse_temp = np.sqrt(np.nanmean((df['model_temp'] - df['temp_mean'])**2))
        rmse_salt = np.sqrt(np.nanmean((df['model_salt'] - df['salt_mean'])**2))
        count = len(df)
        return pd.Series({'rmse_density':rmse_density, 'rmse_temp':rmse_temp, 'rmse_salt':rmse_salt, 'n_obs':count})
    z_min = -60
    dz = 5
    z = np.arange(z_min + dz/2, 0 + dz/2, dz)
    df_rmse = glider_roms_df.groupby(pd.cut(glider_roms_df['z'], np.arange(z_min, 0 + dz, dz))).apply(_rmse).reset_index()
    
    # --- Plot ---
    fig = plt.figure(figsize=(11, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    # --- drho/dx
    ax1 = plt.subplot(2, 2, (1, 2))
    ax1.plot([0, 13], [0, 0], '-k', linewidth=0.5)
    
    obs_mean = df_mahjabin['mean'].values*10**5
    yerr_min = obs_mean - df_mahjabin['min'].values*10**5
    yerr_max = df_mahjabin['max'].values*10**5 - obs_mean
    ax1.errorbar(df_mahjabin['month'].values, obs_mean, yerr=np.array([yerr_min, yerr_max]),
                marker='o', markeredgecolor='k', markerfacecolor='k', ms=5, ecolor='k', linestyle='none',
                label='Glider obs.')
    
    ax1.plot(df_monthly['month'].values, df_monthly['mean'].values*10**5, '-', color=color_dswt, label='CWA-ROMS')
    ax1.fill_between(df_monthly['month'].values, df_monthly['min'].values*10**5, df_monthly['max'].values*10**5, color=color_dswt_std, alpha=0.2)
    
    ax1.set_ylim([-4.0, 4.0])
    ax1.set_xlim([0.5, 12.5])
    ax1.set_xticks(np.arange(1, 13))
    ax1.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax1.set_ylabel(r'$\frac{\partial\rho}{\partial x}$ (10$^{-5}$ kg m$^{-3}$ m${-1}$)')
    ax1.legend(loc='upper right')
    
    add_subtitle(ax1, r'(a) Monthly mean $\frac{\partial\rho}{\partial x}$', location='lower left')
    
    # --- vertical RMSE    
    ax2 = plt.subplot(2, 2, 3)
    ax2.plot(df_rmse['rmse_density'].values, z, '-k', zorder=5, label='Density')
    ax2.plot(df_rmse['rmse_temp'].values, z, '--', color=color_dswt, zorder=4, label='Temperature')
    ax2.plot(df_rmse['rmse_salt'].values, z, ':', color=color_transport, zorder=3, label='Salinity')
    ax2.set_xlim([0, 1.0])
    ax2.set_xlabel('RMSD')
    ax2.set_ylim([-64, -2.5])
    ax2.set_ylabel('Depth (m)')
    
    ax2.legend(loc='lower center')
    
    add_subtitle(ax2, '(b) RMSD vertical profile', location='lower left')
    
    ax22 = ax2.twiny()
    ax22.fill_betweenx(z, 0, df_rmse['n_obs'].values, color=color_dswt_std, alpha=0.2, zorder=2)
    ax22.set_xlabel('Observations (#)')
    ax22.set_xlim([0, 700])
    
    # move axes
    l2, b2, w2, h2 = ax2.get_position().bounds
    ax2.set_position([l2, b2, 1.4*w2, h2])
    
    # --- glider transects
    l2, b2, w2, h2 = ax2.get_position().bounds
    w222 = 0.3*w2
    h222 = 0.2*h2
    dw = 0.01
    dh = 0.2*h2/3
    l2221 = l2+w2+dw
    l2222 = l2+w2+2*dw+w222
    
    for i in range(len(input_files)):
        glider_data = GliderData.read_from_netcdf(f'{glider_dir}{input_files[i]}')
        glider_data.get_data_in_time_frame(start_dates[i], end_dates[i])
        x = glider_data.cumtime
        if i == 0 or i == 2:
            x = -x
        y = -glider_data.depth
        l_depth = y >= -60
        i_depth = np.where(l_depth == True)[0]
        z = glider_data.density - 1000
        z[~l_depth] = np.nan
        
        if i  < 4:
            ax222 = fig.add_axes([l2221, b2+h2-h222-i*(dh+h222), w222, h222])
        else:
            ax222 = fig.add_axes([l2222, b2+h2-(i-3)*h222-(i-4)*dh, w222, h222])
        c222 = ax222.scatter(x, y, c=z, cmap=cm.cm.thermal_r, vmin=24.3, vmax=25.7, s=2, marker='.')
        ax222.fill_between(x, -200, glider_data.z_bottom, color='#d2d2d2', edgecolor='none')
        ax222.set_ylim([-60, 0])
        if i == 0 or i == 2:
            ax222.set_xlim([x[i_depth[-1]], x[i_depth[0]]])
        else:
            ax222.set_xlim([x[i_depth[0]], x[i_depth[-1]]])
        ax222.set_xticks([])
        ax222.set_yticks([])
    _, b222, _, _ = ax222.get_position().bounds
    cax222 = fig.add_axes([l2222, b222-dh-0.02, w222, 0.02])
    cbar222 = plt.colorbar(c222, cax=cax222, orientation='horizontal')
    cbar222.set_label(r'$\sigma_T$ (kg m$^{-3}$)')
    
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()
    
def plot_glider_model_direct_comparisons(output_path='plots/cwa/glider_cwa_direct_comparisons.jpg', show=False):
    glider_roms_df = pd.read_csv('plots/cwa/cwa_glider_roms_comparison.csv')
    
    fig = plt.figure(figsize=(11, 3))
    
    # --- Direct comparisons
    n_samples = len(glider_roms_df)
    vmin = 0
    vmax = 0.01
    
    ax3 = plt.subplot(1, 3, 1)
    min_d = 1024.3 - 1000
    max_d = 1025.8 - 1000
    dd = 0.03
    density_binned, dx_bins, dy_bins = np.histogram2d(glider_roms_df['density_mean'].values-1000, glider_roms_df['model_density'].values-1000,
                                                      bins=np.arange(min_d, max_d + dd, dd))
    density_binned[density_binned == 0] = np.nan
    c = ax3.pcolormesh(dx_bins, dy_bins, density_binned.transpose() / n_samples, cmap='turbo', vmin=vmin, vmax=vmax)
    # ax3.scatter(glider_roms_df['density_mean'].values-1000, glider_roms_df['model_density'].values-1000, marker='x', c='k', s=10)
    ax3.plot([min_d, max_d], [min_d, max_d], '-k')
    ax3.axis('equal')
    ax3.set_xlim([min_d, max_d])
    ax3.set_ylim([min_d, max_d])
    ax3.set_xlabel(r'Glider obs. $\sigma_T$ (kg m$^{-3}$)')
    ax3.set_ylabel(r'CWA-ROMS $\sigma_T$ (kg m$^{-3}$)')
    add_subtitle(ax3, '(a) Model vs. obs. density')
    
    ax4 = plt.subplot(1, 3, 2)
    min_t = 17
    max_t = 23
    dt = 0.1
    temp_binned, tx_bins, ty_bins = np.histogram2d(glider_roms_df['temp_mean'].values, glider_roms_df['model_temp'].values,
                                                    bins=np.arange(min_t, max_t + dt, dt))
    temp_binned[temp_binned == 0] = np.nan
    ax4.pcolormesh(tx_bins, ty_bins, temp_binned.transpose() / n_samples, cmap='turbo', vmin=vmin, vmax=vmax)
    ax4.plot([min_t, max_t], [min_t, max_t], '-k')
    # ax4.scatter(glider_roms_df['temp_mean'].values, glider_roms_df['model_temp'].values, marker='x', c='k', s=10)
    ax4.axis('equal')
    ax4.set_xlim([min_t, max_t])
    ax4.set_ylim([min_t, max_t])
    ax4.set_xlabel('Glider obs. temperature ($^o$C)')
    ax4.set_ylabel('CWA-ROMS temperature ($^o$C)')
    add_subtitle(ax4, '(b) Model vs. obs. temperature')
    
    ax5 = plt.subplot(1, 3, 3)
    min_s = 35.2
    max_s = 36.0
    ds = 0.02
    salt_binned, sx_bins, sy_bins = np.histogram2d(glider_roms_df['salt_mean'].values, glider_roms_df['model_salt'].values,
                                                    bins=np.arange(min_s, max_s + ds, ds))
    salt_binned[salt_binned == 0] = np.nan
    ax5.pcolormesh(sx_bins, sy_bins, salt_binned.transpose() / n_samples, cmap='turbo', vmin=vmin, vmax=vmax)
    ax5.plot([min_s, max_s], [min_s, max_s], '-k')
    # ax5.scatter(glider_roms_df['salt_mean'].values, glider_roms_df['model_salt'].values, marker='x', c='k', s=10)
    ax5.axis('equal')
    ax5.set_xlim([min_s, max_s])
    ax5.set_ylim([min_s, max_s])
    ax5.set_xlabel('Glider obs. salinity')
    ax5.set_ylabel('CWA-ROMS salinity')
    add_subtitle(ax5, '(c) Model vs. obs. salinity')
    
    l5, b5, w5, h5 = ax5.get_position().bounds
    cax = fig.add_axes([l5+w5+0.02, b5, 0.02, h5])
    cbar = plt.colorbar(c, cax=cax)
    cbar.set_label('Samples (fraction)')
    
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()
    
if __name__ == '__main__':
    # ----------------------------------------------------------------------------------
    # Selected Two Rocks glider data with DSWT observations
    # ----------------------------------------------------------------------------------
    input_files = [# may 2010
                   "IMOS_ANFOG_BCEOPSTUV_20100507T030402Z_SL130_FV01_timeseries_END-20100515T133815Z.nc",
                   # july 2010
                   "IMOS_ANFOG_BCEOPSTUV_20100628T045246Z_SL130_FV01_timeseries_END-20100714T233721Z.nc",
                   # sep 2012
                   "IMOS_ANFOG_BCEOPSTUV_20120824T031255Z_SL248_FV01_timeseries_END-20120914T035136Z.nc",
                   # may 2016
                   "IMOS_ANFOG_BCEOPSTUV_20160512T034541Z_SL502_FV01_timeseries_END-20160530T020243Z.nc",
                   # jul 2019
                   "IMOS_ANFOG_BCEOPSTUV_20190625T071857Z_SL248_FV01_timeseries_END-20190723T040704Z.nc",
                   # jul 2020
                   "IMOS_ANFOG_BCEOPSTUV_20200625T074407Z_SL248_FV01_timeseries_END-20200721T053759Z.nc",
                   # jul 2022
                   "IMOS_ANFOG_BCEOPSTUV_20220628T064224Z_SL286_FV01_timeseries_END-20220712T082641Z.nc"]
    
    start_dates = [datetime(2010, 5, 12, 10, 0),
                   datetime(2010, 7, 5, 14, 0),
                   datetime(2012, 9, 10, 0, 0),
                   datetime(2016, 5, 13, 8, 0),
                   datetime(2019, 7, 3, 6, 0),
                   datetime(2020, 7, 5),
                   datetime(2022, 7, 5, 0, 0)]
    end_dates = [datetime(2010, 5, 14, 0, 0),
                 datetime(2010, 7, 8, 0, 0),
                 datetime(2012, 9, 11, 22, 0),
                 datetime(2016, 5, 15, 0, 0),
                 datetime(2019, 7, 6, 0, 0),
                 datetime(2020, 7, 6, 10, 0),
                 datetime(2022, 7, 6, 5, 0)]
    flip = [True, False, True, False, False, False, False, False]
    
    # # ----------------------------------------------------------------------------------
    # # Calculate and write drho/dx ROMS data to csv - can skip if already exists
    # # ----------------------------------------------------------------------------------
    # tr_lon_range, tr_lat_range = get_approximate_two_rocks_coordinates(input_files, start_dates, end_dates)
    # # select transects in vicinity to Two Rocks transect
    # dx = 0.2
    # dy = 0.2
    # transects = read_transects_in_lon_lat_range_from_json('input/transects/cwa_transects.json',
    #                                                       [tr_lon_range[0]-dx, tr_lon_range[1]+dx],
    #                                                       [tr_lat_range[0]-dy, tr_lat_range[1]+dy])
    # # get drho/dx along Two Rocks transects and write to file
    # years = np.arange(2008, 2019)
    # calculate_drhodx_model(transects, years, config)
    
    # # ----------------------------------------------------------------------------------
    # # Compare vertical profile data and write to csv - skip is already exists
    # # ----------------------------------------------------------------------------------
    # compare_glider_model_vertical_profiles(input_files, start_dates, end_dates)
    
    # ----------------------------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------------------------
    plot_glider_model_comparisons(input_files, start_dates, end_dates)
    plot_glider_model_direct_comparisons()
        