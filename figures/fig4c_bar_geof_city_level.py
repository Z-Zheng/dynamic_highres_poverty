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

df = df[df['model'].isin(['Transformer', 'Transformer_geof',
                          ])]
df = df[df['site'].isin(['li', 'bl'])]

df.loc[df['model'] == 'Transformer', 'model'] = 'satellite image only'
df.loc[df['model'] == 'Transformer_geof', 'model'] = 'satellite image w/ geo-features'

df.loc[df['model'] == 'XGB-w/-geof', 'frac'] *= 100
df.loc[df['model'] == 'XGB-w/-geof_10hh', 'frac'] *= 100

df.loc[df['model'] == 'XGB-w/-geof', 'model'] = 'XGboost / full household'
df.loc[df['model'] == 'XGB-w/-geof_10hh', 'model'] = 'XGboost / 10-household'

frac = [1, 5, 10, 25, 50, 100]

df = df[df['frac'].isin(frac)]

col_order = ['li', 'bl', ]
g = sns.catplot(
    df, x='frac', y='r2', hue='model', col='site', col_order=col_order,
    col_wrap=2, kind='bar', height=4, aspect=1,
    sharex=False,
    palette='Paired',
)

g._legend.set_bbox_to_anchor((0.28, 0.87))
g._legend.set_title('')

g.set_axis_labels(x_var='#traininig data (fraction)', y_var='R$^2$')
MAPPING = {
    'mi_08': 'Level | Malawi (2008)', 'mi_18': 'Level | Malawi (2018)', 'mz_07': 'Level | Mozambique (2007)',
    'mz_17': 'Level | Mozambique (2017)', 'mg': 'Level | Madagascar (2018)', 'bf': 'Level | Burkina Faso (2019)',
    'li': 'City | Lilongwe (2023)', 'bl': 'City | Blantyre (2023)',
    'mi_08_18': 'Change | Malawi (2008-2018)',
    'mz_07_17': 'Change | Mozambique (2007-2017)',
}
N_IMAGES = {'mi_08': 2745, 'mi_18': 2745, 'mi_08_18': 2745,
            'mz_07': 23268, 'mz_17': 23268, 'mz_07_17': 23268,
            'mg': 15612, 'bf': 10512, 'li': 1915, 'bl': 1551}

for ax, title in zip(g.axes.flat, g.col_names):
    ax.set_title(MAPPING[title])
    new_xticks = [f'{int(N_IMAGES[title] * f / 100)} ({f}%)' for f in frac]
    ax.set_xticklabels(
        new_xticks,
        rotation=30
    )

plt.tight_layout()

plt.show()