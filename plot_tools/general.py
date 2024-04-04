import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

def add_subtitle(ax:plt.axes, subtitle:str, location='upper left') -> plt.axes:
    anchored_text = AnchoredText(subtitle, loc=location, borderpad=0.0)
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