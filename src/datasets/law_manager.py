"""Law Data Manager."""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss

from src.datasets.abstract_manager import AbstractManager
from src.util.cleaning import FeatureInfo, clean_dataframe
from src.util.plot import ColorFader
from src.util.preprocessing import split_dataset
from src.util.typing import Methods, Augmented, SamplingFunctions, Rng, Figsize, TightLayout


class LawManager(AbstractManager):
    """Data Manager for the Law Dataset."""

    # https://rdrr.io/cran/fairml/man/law.school.admissions.html
    FEATURES: Dict[str, FeatureInfo] = {
        'pass_bar': FeatureInfo(kind='float', alias='pass'),
        'lsat': FeatureInfo(kind='float', alias='lsat'),
        'ugpa': FeatureInfo(kind='float', alias='ugpa'),
        'decile1': FeatureInfo(kind='float', alias='first_year_decile'),
        'decile3': FeatureInfo(kind='float', alias='third_year_decile'),
        'fam_inc': FeatureInfo(kind='float', alias='family_income'),
        'gender': FeatureInfo(kind='category', alias='gender'),
        'race1': FeatureInfo(kind='category', alias='race'),
        'cluster': FeatureInfo(kind='category', alias='prestige'),
        'fulltime': FeatureInfo(kind='category', alias='fulltime')
    }

    # noinspection PyMissingOrEmptyDocstring
    @staticmethod
    def load_data(filepath: str, full_features: bool, train_fraction: float) -> AbstractManager.Data:
        df = pd.read_csv(filepath)
        df = clean_dataframe(df, LawManager.FEATURES)
        if full_features:
            df = pd.get_dummies(df, prefix_sep='/').dropna()
        else:
            df = df[['lsat', 'ugpa', 'pass']].dropna()
        return split_dataset(df, test_size=1 - train_fraction, val_size=0.0, stratify=df['pass'])

    def __init__(self, filepath: str, full_features: bool = False, full_grid: bool = False, grid_augmented: int = 8,
                 grid_ground: Optional[int] = None, x_scaling: Methods = 'std', train_fraction: float = 0.8):
        grid = None
        if full_features:
            assert full_grid is False, "'full_grid' is not supported with 'full_features'"
            x_scaling = {v.alias or k: x_scaling for k, v in LawManager.FEATURES.items() if v.kind == 'float'}
        elif full_grid:
            lsat, ugpa = np.meshgrid(np.linspace(0, 50, 64), np.linspace(0, 4, 64))
            grid = pd.DataFrame.from_dict({'lsat': lsat.flatten(), 'ugpa': ugpa.flatten()})
            x_scaling = {'lsat': x_scaling, 'ugpa': x_scaling}
        super(LawManager, self).__init__(
            directions={'lsat': 1, 'ugpa': 1},
            stratify=True,
            x_scaling=x_scaling,
            y_scaling=None,
            label='pass',
            loss=log_loss,
            loss_name='bce',
            metric=accuracy_score,
            metric_name='acc',
            post_process=lambda x: x.round().astype(int),
            data_kwargs=dict(figsize=(14, 8), tight_layout=True),
            augmented_kwargs=dict(figsize=(10, 4), tight_layout=True),
            summary_kwargs=dict(figsize=(14, 4), tight_layout=True, res=50),
            grid_kwargs=dict(num_augmented=grid_augmented, num_ground=grid_ground),
            grid=grid,
            filepath=filepath,
            full_features=full_features,
            train_fraction=train_fraction
        )

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
