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

df = df[df['model'].isin([
    'CNN',
    'Transformer',
    'XGB-w/o-geof',
])]
df = df[df['site'].isin(['mi_08_18', 'mz_07_17'])]

df.loc[df['model'] == 'XGB-w/o-geof', 'frac'] *= 100
df.loc[df['model'] == 'XGB-w/o-geof', 'model'] = 'XGBoost w/ sat-features'

df.loc[df['model'] == 'XGB-only-w/-geof', 'frac'] *= 100
df.loc[df['model'] == 'XGB-only-w/-geof', 'model'] = 'XGBoost w/ geo-features'

df.loc[df['model'] == 'XGB-w/-geof', 'frac'] *= 100
df.loc[df['model'] == 'XGB-w/-geof', 'model'] = 'XGBoost w/ sat + geo-features'

g = sns.relplot(
    df, x='frac', y='r2', hue='model', col='site',
    col_wrap=2, kind='line', height=4, aspect=1,
    col_order=['mi_08_18', 'mz_07_17', ],
    hue_order=['CNN', 'Transformer', 'XGBoost w/ sat-features'],
    markers=True,
    style='model',
    facet_kws=dict(sharex=False, sharey=True)
)

g._legend.set_bbox_to_anchor((0.26, 0.84))

g.set_axis_labels(x_var='#traininig data (fraction)', y_var='R$^2$')
MAPPING = {
    'mi_08': 'Country | Malawi (2008)', 'mi_18': 'Country | Malawi (2018)', 'mz_07': 'Country | Mozambique (2007)',
    'mz_17': 'Country | Mozambique (2017)', 'mg': 'Country | Madagascar (2018)', 'bf': 'Country | Burkina Faso (2019)',
    'li': 'City | Lilongwe (2023)', 'bl': 'City | Blantyre (2023)',
    'mi_08_18': 'Country | Malawi (2008-2018)',
    'mz_07_17': 'Country | Mozambique (2007-2017)'
}
N_IMAGES = {'mi_08': 2745, 'mi_18': 2745, 'mi_08_18': 2745,
            'mz_07': 23268, 'mz_17': 23268, 'mz_07_17': 23268,
            'mg': 15612, 'bf': 10512, 'li': 1915, 'bl': 1551}

g.axes.flat[0].set_ylim(0, )

for ax, title in zip(g.axes.flat, g.col_names):
    ax.set_title(MAPPING[title])

    new_xticks = [f'{int(N_IMAGES[title] * f / 100)} ({f}%)' for f in [1, 5, 10, 25, 50, 100]]
    ax.set_xticks([1, 5, 10, 25, 50, 100])
    ax.set_xticklabels(
        new_xticks,
        rotation=90
    )

plt.tight_layout()

plt.show()
