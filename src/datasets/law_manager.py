from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from src.datasets.data_manager import DataManager
from src.util.plot import ColorFader
from src.util.preprocessing import split_dataset
from src.util.augmentation import compute_numeric_monotonicities


class LawManager(DataManager):
    def __init__(self, filepath: str, x_scaling: Any = 'std', y_scaling: Any = 'norm', test_size: float = 0.8,
                 res: int = 64):
        self.filepath: str = filepath
        self.test_size: float = test_size
        lsat, ugpa = np.meshgrid(np.linspace(0, 50, res), np.linspace(0, 4, res))
        super(LawManager, self).__init__(
            x_columns=['lsat', 'ugpa'],
            x_scaling=x_scaling,
            y_column='pass',
            y_scaling=y_scaling,
            metric=accuracy_score,
            metric_name='acc',
            post_process=lambda x: x.round().astype(int),
            grid=pd.DataFrame.from_dict({'lsat': lsat.flatten(), 'ugpa': ugpa.flatten()}),
            data_kwargs=dict(figsize=(14, 8), tight_layout=True),
            augmented_kwargs=dict(figsize=(10, 4), tight_layout=True),
            summary_kwargs=dict(figsize=(14, 4), tight_layout=True, res=50)
        )

    def compute_monotonicities(self, samples, references, eps=1e-5):
        return compute_numeric_monotonicities(samples, references, directions=[1, 1], eps=eps)

    def _load_splits(self):
        # preprocess data
        df = pd.read_csv(self.filepath)[['lsat', 'ugpa', 'pass_bar']]
        df = df.dropna().reset_index(drop=True)
        df = df.rename(columns={'pass_bar': 'pass'}).astype({'pass': int})
        # split data
        return split_dataset(df[['lsat', 'ugpa']], df['pass'], test_size=self.test_size, val_size=0.5, random_state=0)

    def _get_sampling_functions(self, num_augmented, rng):
        return {
            'lsat': (num_augmented // 2, lambda s: rng.uniform(0, 50, size=s)),
            'ugpa': (num_augmented // 2, lambda s: rng.uniform(0, 4, size=s))
        }

    def _data_plot(self, figsize, tight_layout, **kwargs):
        _, ax = plt.subplots(2, len(kwargs), sharex='col', sharey='col', figsize=figsize, tight_layout=tight_layout)
        for i, (title, (x, y)) in enumerate(kwargs.items()):
            for c, label in enumerate(['Passed', 'Not Passed']):
                data = x[y == 1 - c]
                sns.kdeplot(x='ugpa', y='lsat', data=data, fill=True, ax=ax[c, i])
                sns.scatterplot(x='ugpa', y='lsat', data=data, s=25, alpha=0.7, marker='+', color='black', ax=ax[c, i])
                ax[c, i].set(title=title.capitalize(), ylabel=label if i == 0 else None, xlim=(1.4, 4.3), ylim=(0, 52))

    # noinspection PyMethodOverriding
    def _summary_plot(self, model, res, figsize, tight_layout, **kwargs):
        lsat, ugpa = np.meshgrid(np.linspace(0, 50, res), np.linspace(0, 4, res))
        grid = pd.DataFrame.from_dict({'lsat': lsat.flatten(), 'ugpa': ugpa.flatten()})
        grid['pred'] = model.predict(grid)
        _, axes = plt.subplots(1, 3, figsize=figsize, tight_layout=tight_layout)
        axes[0].pcolor(lsat, ugpa, grid['pred'].values.reshape(res, res), shading='auto', vmin=0, vmax=1)
        axes[0].set(xlabel='lsat', ylabel='ugpa')
        for ax, (feat, group_feat) in zip(axes[1:], [('lsat', 'ugpa'), ('ugpa', 'lsat')]):
            lb, ub = grid[group_feat].min(), grid[group_feat].max()
            fader = ColorFader('red', 'blue', bounds=(lb, ub))
            for group_val, group in grid.groupby([group_feat]):
                label = f'{group_feat}: {group_val:.0f}' if (group_val in [lb, ub]) else None
                sns.lineplot(data=group, x=feat, y='pred', color=fader(group_val), alpha=0.6, label=label, ax=ax)
