# Copyright (c) Stanford Sustainability and AI Lab and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme(
    context='paper',
    style='ticks',
    font="sans-serif"
)

df = pd.read_csv('./data/wb_results_all.csv')

df = df[df['model'].isin(['Transformer', ])]
df = df[df['site'].isin(['li', 'li_planetscope', 'bl', 'bl_planetscope'])]

df.loc[df['site'] == 'li', 'sensor'] = 'SkySat (0.5m)'
df.loc[df['site'] == 'li_planetscope', 'sensor'] = 'PlanetScope (3m)'
df.loc[df['site'] == 'li_planetscope', 'site'] = 'li'
df.loc[df['site'] == 'li', 'site'] = 'li'

df.loc[df['site'] == 'bl', 'sensor'] = 'SkySat (0.5m)'
df.loc[df['site'] == 'bl_planetscope', 'sensor'] = 'PlanetScope (3m)'
df.loc[df['site'] == 'bl_planetscope', 'site'] = 'bl'
df.loc[df['site'] == 'bl', 'site'] = 'bl'


cmap = sns.color_palette('tab10')

g = sns.catplot(
    df, x='frac', y='r2', hue='sensor',
    col='site',
    col_wrap=2, kind='bar', height=4, aspect=1,
    palette=[cmap[3], cmap[-3]],
    sharex=False,
)

g._legend.set_bbox_to_anchor((0.23, 0.85))
g.set_axis_labels(x_var='#traininig data (fraction)', y_var='R$^2$')

MAPPING = {
    'mi_08': 'Country | Malawi (2008)', 'mi_18': 'Country | Malawi (2018)', 'mz_07': 'Country | Mozambique (2007)',
    'mz_17': 'Country | Mozambique (2017)', 'mg': 'Country | Madagascar (2018)', 'bf': 'Country | Burkina Faso (2019)',
    'li': 'City | Lilongwe (2023)', 'bl': 'City | Blantyre (2023)'
}
N_IMAGES = {'mi_08': 2745, 'mi_18': 2745, 'mi_08_18': 2745,
            'mz_07': 23268, 'mz_17': 23268, 'mz_07_17': 23268,
            'mg': 15612, 'bf': 10512, 'li': 1915, 'bl': 1551}

for ax, title in zip(g.axes.flat, g.col_names):
    ax.set_title(MAPPING[title])

    new_xticks = [f'{int(N_IMAGES[title] * f / 100)} ({f}%)' for f in [1, 5, 10, 25, 50, 100]]

    ax.set_xticklabels(
        new_xticks,
        rotation=30
    )

plt.tight_layout()
plt.show()
