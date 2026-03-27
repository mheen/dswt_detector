import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np

def add_subtitle(ax:plt.axes, subtitle:str, location='upper left', alpha=1.0) -> plt.axes:
    anchored_text = AnchoredText(subtitle, loc=location, borderpad=0.0)
    anchored_text.patch.set_alpha(alpha)
    anchored_text.zorder = 15
    ax.add_artist(anchored_text)
    return ax

def color_y_axis(ax:plt.axes, color:str, spine_location:str):
    ax.spines[spine_location].set_color(color)
    ax.tick_params(axis='y', colors=color)
    ax.yaxis.label.set_color(color)
    return ax

def add_wind_dir_ticks(ax:plt.axes) -> plt.axes:
    yticks = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    ytick_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    return ax

def get_vmin_vmax(values:np.ndarray[float], min_bin=1024.0, max_bin=1027.0, dbin=0.1):
    bins = np.arange(min_bin, max_bin, dbin)
    bin_edges = np.empty(len(bins)+1)
    bin_edges[:-1] = bins - dbin/2
    bin_edges[-1] = bins[-1] + dbin/2
    
    n, _ = np.histogram(values[~np.isnan(values)], bins=bin_edges)
    bins_most_values = bins[n >= 0.2 * np.nanmax(n)]
    
    vmin = bins_most_values[0]
    vmax = bins_most_values[-1]
    return vmin, vmax