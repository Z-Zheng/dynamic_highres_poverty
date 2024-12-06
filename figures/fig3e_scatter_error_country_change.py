# Copyright (c) Stanford Sustainability and AI Lab and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import matplotlib.pyplot as plt
import math
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score

gt, pr = np.load('./data/gt_pr_mi_c_Transformer/0_gt_pr.npy')

r2 = r2_score(gt, pr)
print(f'R2: {r2:.2f}')

minv = math.floor(np.min(gt))
maxv = math.ceil(np.max(gt))
sns.set(style="ticks")

g = sns.JointGrid()
sns.scatterplot(x=gt, y=pr, s=5, legend=False, alpha=0.3, edgecolor=None, color='black', ax=g.ax_joint)
g.ax_joint.set_title('Country | Malawi (2008-2018)')

cmap = sns.color_palette('tab10')
sns.kdeplot(x=gt, linewidth=2, ax=g.ax_marg_x, color=cmap[3])
sns.kdeplot(y=pr, linewidth=2, ax=g.ax_marg_y, color=cmap[3])

g.ax_joint.text(0.05, 0.95, f'R$^2$: {r2:.2f}', horizontalalignment='left', verticalalignment='center',
                transform=g.ax_joint.transAxes, fontsize=plt.rcParams['font.size'])

x_line = np.linspace(minv, maxv, 100)
g.ax_joint.plot(x_line, x_line, color='black', linestyle='--', label='y=x')

g.ax_joint.set_xlim(minv, maxv)
g.ax_joint.set_ylim(minv, maxv)
g.ax_joint.set_xlabel('Survey-measured asset wealth change')
g.ax_joint.set_ylabel('Model-predicted asset wealth change')

sns.despine(ax=g.ax_joint)
sns.despine(ax=g.ax_marg_x, left=True)
sns.despine(ax=g.ax_marg_y, bottom=True)
plt.tight_layout()
plt.show()
