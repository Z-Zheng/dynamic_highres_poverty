# Copyright (c) Stanford Sustainability and AI Lab and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.ticker import MaxNLocator

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig = plt.figure(
    constrained_layout=True,
    figsize=(8, 8)
)
gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 1,], height_ratios=[1, 1,])
ax_mi = fig.add_subplot(gs[0, 0], projection=ccrs.Mercator())
ax_mz = fig.add_subplot(gs[0, 1], projection=ccrs.Mercator())
ax_mg = fig.add_subplot(gs[1, 0], projection=ccrs.Mercator())
ax_bf = fig.add_subplot(gs[1, 1], projection=ccrs.Mercator())

cmap = 'inferno'
gdf_mi = gpd.read_file('./data/full_pred_mi_18.geojson').to_crs('EPSG:3857')
gdf_mi.plot(column='awi', cmap=cmap, legend=True, edgecolor='none', ax=ax_mi,
            legend_kwds={"orientation": "horizontal"})
ax_mi.set_title('Country | Malawi (2018)', fontdict={'family': 'sans-serif', 'size': plt.rcParams['font.size']})

gdf_mz = gpd.read_file('./data/full_pred_mz_17.geojson').to_crs('EPSG:3857')
gdf_mz.plot(column='awi', cmap=cmap, legend=True, edgecolor='none', ax=ax_mz,
            legend_kwds={"orientation": "horizontal"})
ax_mz.set_title('Country | Mozambique (2017)', fontdict={'family': 'sans-serif', 'size': plt.rcParams['font.size']})

gdf_mg = gpd.read_file('./data/full_pred_mg_18.geojson').to_crs('EPSG:3857')
gdf_mg.plot(column='awi', cmap=cmap, legend=True, edgecolor='none', ax=ax_mg,
            legend_kwds={"orientation": "horizontal"})
ax_mg.set_title('Country | Madagascar (2018)', fontdict={'family': 'sans-serif', 'size': plt.rcParams['font.size']})

gdf_bf = gpd.read_file('./data/full_pred_bf_19.geojson').to_crs('EPSG:3857')
gdf_bf.plot(column='awi', cmap=cmap, legend=True, edgecolor='none', ax=ax_bf,
            legend_kwds={"orientation": "horizontal"})
ax_bf.set_title('Country | Burkina Faso (2019)', fontdict={'family': 'sans-serif', 'size': plt.rcParams['font.size']})





for ax in [ax_mi, ax_mz, ax_bf, ax_mg]:
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
