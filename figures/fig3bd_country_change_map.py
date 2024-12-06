# Copyright (c) Stanford Sustainability and AI Lab and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig = plt.figure(
    constrained_layout=True,
    figsize=(4, 8)
)
gs = gridspec.GridSpec(2, 1, figure=fig, width_ratios=[1, ], height_ratios=[1, 1, ])
ax_mi = fig.add_subplot(gs[0, 0], projection=ccrs.Mercator())
ax_mz = fig.add_subplot(gs[1, 0], projection=ccrs.Mercator())

cmap = plt.get_cmap('RdBu_r')
norm = mcolors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=4)
gdf_mi = gpd.read_file('./data/full_pred_mi_c.geojson').to_crs('EPSG:3857')
gdf_mi.plot(column='awi', cmap=cmap, legend=True, edgecolor='none', ax=ax_mi, norm=norm,
            legend_kwds={"orientation": "horizontal"})
ax_mi.set_title('Country | Malawi (2008-2018)', fontdict={'family': 'sans-serif', 'size': plt.rcParams['font.size']})

gdf_mz = gpd.read_file('./data/full_pred_mz_c.geojson').to_crs('EPSG:3857')
gdf_mz.plot(column='awi', cmap=cmap, legend=True, edgecolor='none', ax=ax_mz, norm=norm,
            legend_kwds={"orientation": "horizontal"})
ax_mz.set_title('Country | Mozambique (2007-2017)', fontdict={'family': 'sans-serif', 'size': plt.rcParams['font.size']})

for ax in [ax_mi, ax_mz, ]:
    ax.set_axis_off()

    gl = ax.gridlines(
        draw_labels={"top": "x", "right": "y"},
        color='gray',
        alpha=0.5,
        linewidth=2,
        xpadding=1,
        ypadding=1,
    )
    gl.xlines = None
    gl.ylines = None
    gl.ylabel_style = {'rotation': 90, 'color': 'black', 'weight': 'bold'}
    gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
    gl.xlocator = MaxNLocator(nbins=3)
    gl.ylocator = MaxNLocator(nbins=3)

plt.show()
