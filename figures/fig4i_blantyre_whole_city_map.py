# Copyright (c) Stanford Sustainability and AI Lab and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
import cartopy.crs as ccrs
from matplotlib.ticker import MaxNLocator

fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={
    'projection': ccrs.Mercator()
})

cmap = 'RdBu_r'

gdf = gpd.read_file('./data/Blantyre_pred_grid_300x300.geojson').to_crs('EPSG:3857')
gdf.plot(column='awi', cmap=cmap, legend=True, edgecolor='none', ax=ax)

gl = ax.gridlines(
    draw_labels={"top": "x", "right": "y"},
    color='gray',
    alpha=0.5,
    linewidth=2,
    xpadding=-5,
    ypadding=-5,
)
gl.xlines = None
gl.ylines = None
gl.ylabel_style = {'rotation': 90, 'color': 'black', 'weight': 'bold'}
gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
gl.xlocator = MaxNLocator(nbins=3)
gl.ylocator = MaxNLocator(nbins=3)

plt.show()
