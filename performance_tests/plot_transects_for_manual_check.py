from plot_tools.general import add_subtitle
import matplotlib.pyplot as plt
import cmocean as cm
import xarray as xr
import pandas as pd
import numpy as np

cmap_density = cm.cm.thermal_r
cmap_temp = 'RdYlBu_r'
cmap_salt = cm.cm.haline

def _get_vmin_vmax(values:np.ndarray[float]):
    min_value = np.nanmin(values)
    max_value = np.nanmax(values)
    dvalue = max_value - min_value
    
    dbin = dvalue/20
    
    bins = np.arange(min_value, max_value + dbin, dbin)
    
    bin_edges = np.empty(len(bins)+1)
    bin_edges[:-1] = bins - dbin/2
    bin_edges[-1] = bins[-1] + dbin/2
    
    n, _ = np.histogram(values[~np.isnan(values)], bins=bin_edges)
    bins_most_values = bins[n >= 0.2 * np.nanmax(n)]
    
    vmin = bins_most_values[0]
    vmax = bins_most_values[-1]
    return vmin, vmax

def _plot_transect(transect_ds:xr.Dataset, ax:plt.axes, variable:str, t_dswt:int, cmap:str) -> tuple:
    vmin, vmax = _get_vmin_vmax(transect_ds[variable].values[t_dswt, :, :])
    
    c = transect_ds[variable][t_dswt, :, :].plot(x='distance', y='z_rho', vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=False)
    ax.fill_between(transect_ds.distance.values, -210, -transect_ds.h.values, color='#d2d2d2', edgecolor='k')
    ax.set_xlim([transect_ds.distance.values[0], transect_ds.distance.values[-1]])
    ax.set_ylim([-200, 0])
    ax.set_title('')
    ax.set_ylabel('Depth (m)')
    ax.set_yticks([0, -50, -100, -150, -200])
    ax.set_yticklabels([0, 50, 100, 150, 200])
    
    ax.set_xticks([])
    ax.set_xlabel('')
    
    return c, ax

def _add_colorbar(fig:plt.figure, ax:plt.axes, c, label:str):
    l, b, w, h = ax.get_position().bounds
    cbax = fig.add_axes([l+w+0.02, b, 0.02, h])
    cbar = plt.colorbar(c, cax=cbax)
    cbar.set_label(label)

def transects_plot(transect_ds:xr.Dataset, t_dswt:int) -> plt.figure:
    
    fig = plt.figure(figsize=(6, 8))
      
    # --- Transect density
    ax1 = plt.subplot(3, 1, 1)
    c1, _ = _plot_transect(transect_ds, ax1, 'density', t_dswt, cmap_density)
    add_subtitle(ax1, f'Density along transect', location='lower left')
    _add_colorbar(fig, ax1, c1, 'Density (kg/m$^3$)')
    
    # --- Transect temperature
    ax2 = plt.subplot(3, 1, 2)
    c2, _ = _plot_transect(transect_ds, ax2, 'temp', t_dswt, cmap_temp)
    add_subtitle(ax2, f'Temperature along transect', location='lower left')
    _add_colorbar(fig, ax2, c2, 'Temperature ($^o$C)')
    
    # --- Transect salinity
    ax3 = plt.subplot(3, 1, 3)
    c3, _ = _plot_transect(transect_ds, ax3, 'salt', t_dswt, cmap_salt)
    add_subtitle(ax3, f'Salinity along transect', location='lower left')
    _add_colorbar(fig, ax3, c3, 'Salinity')
    
    # title
    ax1.set_title(f'{pd.to_datetime(transect_ds.ocean_time[t_dswt].values).strftime("%d-%m-%Y %H:%M")}')
    
    return fig