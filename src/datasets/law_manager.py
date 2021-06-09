"""Law Data Manager."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss

from moving_targets.util.typing import Splits
from src.datasets.abstract_manager import AbstractManager
from src.util.plot import ColorFader
from src.util.preprocessing import split_dataset
from src.util.typing import Methods, Augmented, SamplingFunctions, Rng, Figsize, TightLayout


class LawManager(AbstractManager):
    """Data Manager for the Law Dataset."""

    def __init__(self, filepath: str, x_scaling: Methods = 'std', y_scaling: Methods = 'norm',
                 train_fraction: float = 0.03, res: int = 64):
        self.filepath: str = filepath
        self.train_fraction: float = train_fraction
        lsat, ugpa = np.meshgrid(np.linspace(0, 50, res), np.linspace(0, 4, res))
        super(LawManager, self).__init__(
            x_columns=['lsat', 'ugpa'],
            x_scaling=x_scaling,
            y_column='pass',
            y_scaling=y_scaling,
            directions=[1, 1],
            loss=log_loss,
            loss_name='bce',
            metric=accuracy_score,
            metric_name='acc',
            post_process=lambda x: x.round().astype(int),
            grid=pd.DataFrame.from_dict({'lsat': lsat.flatten(), 'ugpa': ugpa.flatten()}),
            data_kwargs=dict(figsize=(14, 8), tight_layout=True),
            augmented_kwargs=dict(figsize=(10, 4), tight_layout=True),
            summary_kwargs=dict(figsize=(14, 4), tight_layout=True, res=50)
        )

    def _load_splits(self, num_folds: Optional[int], extrapolation: bool) -> Splits:
        assert extrapolation is False, "'extrapolation' is not supported for Law dataset"
        # preprocess data
        df = pd.read_csv(self.filepath)[['lsat', 'ugpa', 'pass_bar']]
        df = df.dropna().reset_index(drop=True)
        df = df.rename(columns={'pass_bar': 'pass'}).astype({'pass': int})
        x, y = df[['lsat', 'ugpa']], df['pass']
        # split train/test
        splits = split_dataset(x, y, test_size=1 - self.train_fraction, val_size=0.0, stratify=y)
        return self.cross_validate(splits=splits, num_folds=num_folds, stratify=True)

    def _get_sampling_functions(self, rng: Rng, num_augmented: Augmented = 5) -> SamplingFunctions:
        return {
            'lsat': (num_augmented, lambda s: rng.uniform(0, 50, size=s)),
            'ugpa': (num_augmented, lambda s: rng.uniform(0, 4, size=s))
        }

    def _data_plot(self, figsize: Figsize, tight_layout: TightLayout, **kwargs):
        _, ax = plt.subplots(2, len(kwargs), sharex='col', sharey='col', figsize=figsize, tight_layout=tight_layout)
        for i, (title, (x, y)) in enumerate(kwargs.items()):
            for c, label in enumerate(['Passed', 'Not Passed']):
                data = x[y == 1 - c]
                sns.kdeplot(x='ugpa', y='lsat', data=data, fill=True, ax=ax[c, i])
                sns.scatterplot(x='ugpa', y='lsat', data=data, s=25, alpha=0.7, marker='+', color='black', ax=ax[c, i])
                ax[c, i].set(title=title.capitalize(), ylabel=label if i == 0 else None, xlim=(1.4, 4.3), ylim=(0, 52))

    def _summary_plot(self, model, figsize: Figsize, tight_layout: TightLayout, **kwargs):
        res = kwargs.pop('res')
        lsat, ugpa = np.meshgrid(np.linspace(0, 50, res), np.linspace(0, 4, res))
        grid = pd.DataFrame.from_dict({'lsat': lsat.flatten(), 'ugpa': ugpa.flatten()})
        grid['pred'] = model.predict(grid)
        _, axes = plt.subplots(1, 3, figsize=figsize, tight_layout=tight_layout)
        axes[0].pcolor(lsat, ugpa, grid['pred'].values.reshape(res, res), shading='auto', vmin=0, vmax=1)
        axes[0].set(xlabel='lsat', ylabel='ugpa')
        for ax, (feat, group_feat) in zip(axes[1:], [('lsat', 'ugpa'), ('ugpa', 'lsat')]):
            lb, ub = grid[group_feat].min(), grid[group_feat].max()
            fader = ColorFader('red', 'blue', bounds=[lb, ub])
            for group_val, group in grid.groupby([group_feat]):
                label = f'{group_feat}: {group_val:.0f}' if (group_val in [lb, ub]) else None
                sns.lineplot(data=group, x=feat, y='pred', color=fader(group_val), alpha=0.6, label=label, ax=ax)
