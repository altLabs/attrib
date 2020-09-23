"""
Utilities for plotting beautiful figures.
"""
import matplotlib.pyplot as plt
import palettable as pal
import seaborn as sns
import pandas as pd

def prettify_ax(ax):
    """
    Nifty function we can use to make our axes more pleasant to look at
    """
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_frameon = True
    ax.patch.set_facecolor('#eeeeef')
    ax.grid('on', color='w', linestyle='-', linewidth=1)
    ax.tick_params(direction='out')
    ax.set_axisbelow(True)


def simple_ax(figsize=(6, 4), **kwargs):
    """
    Shortcut to make and 'prettify' a simple figure with 1 axis
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, **kwargs)
    prettify_ax(ax)
    return fig, ax

def set_pub_plot_context(colors='categorical', context="talk"):
	if colors == 'categorical':
		palette = tuple(pal.cartocolors.qualitative.Safe_10.mpl_colors)
	if colors == "sequential":
		palette = tuple(pal.colorbrewer.sequential.YlOrBr_9.mpl_colors)
	sns.set(
		palette=palette,
		style="whitegrid",
		context=context,
            font="Arial"
	)


def save_for_pub(fig, path="../../data/default", dpi=1000):
    fig.savefig(path + ".png", dpi=dpi, bbox_inches='tight')
    fig.savefig(path + ".eps", dpi=dpi, bbox_inches='tight')
    fig.savefig(path + ".pdf", dpi=dpi, bbox_inches='tight')
    fig.savefig(path + ".svg", dpi=dpi, bbox_inches='tight')
    # fig.savefig(path + ".tif", dpi=dpi)


def label_point(x, y, val, ax, fontsize=20, rotation=0):
	"""
	Label x, y points on a given ax with val text of a particular 
	fontsize
	"""
	a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
	for i, point in a.iterrows():
		ax.text(point['x']+.02, point['y'], str(point['val']),
               fontsize=fontsize, rotation=rotation)
       	
