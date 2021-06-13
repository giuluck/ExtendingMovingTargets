"""Default Data Manager."""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss

from src.datasets.abstract_manager import AbstractManager
from src.util.cleaning import FeatureInfo, clean_dataframe
from src.util.preprocessing import split_dataset
from src.util.typing import Augmented, SamplingFunctions, Rng, Figsize, TightLayout


class DefaultManager(AbstractManager):
    """Data Manager for the Default Dataset."""

    MARKERS = {k: v for k, v in enumerate(['o', 's', '^', '+'])}
    FEATURES: Dict[str, FeatureInfo] = {
        'MARRIAGE': FeatureInfo(kind='float', alias='married'),
        'PAY_0': FeatureInfo(kind='float', alias='payment'),
        'default': FeatureInfo(kind='float', alias=None)
    }

    # noinspection PyMissingOrEmptyDocstring
    @staticmethod
    def load_data(filepath: str, train_fraction: float) -> AbstractManager.Data:
        df = pd.read_csv(filepath)
        df = clean_dataframe(df, DefaultManager.FEATURES)
        df = df.dropna().reset_index(drop=True)
        df = df[np.in1d(df['married'], [1, 2])]
        df['married'] = df['married'] - 1
        return split_dataset(df, test_size=1 - train_fraction, val_size=0.0, stratify=df['married'])

    def __init__(self, filepath: str, full_features: bool = False, full_grid: bool = False,
                 grid_ground: Optional[int] = None, x_scaling: str = 'std', train_fraction: float = 0.025):
        grid = None
        if full_features:
            assert full_grid is False, "'full_grid' is not supported with 'full_features'"
        elif full_grid:
            married, payment = np.meshgrid([0, 1], np.arange(-2, 9))
            grid = pd.DataFrame.from_dict({'married': married.flatten(), 'payment': payment.flatten()})
        super(DefaultManager, self).__init__(
            directions={'married': 0, 'payment': 1},
            stratify=True,
            x_scaling={'payment': x_scaling},
            y_scaling=None,
            label='default',
            loss=log_loss,
            loss_name='bce',
            metric=accuracy_score,
            metric_name='acc',
            post_process=lambda x: x.round().astype(int),
            data_kwargs=dict(figsize=(12, 10), tight_layout=True),
            augmented_kwargs=dict(figsize=(10, 4), tight_layout=True),
            summary_kwargs=dict(figsize=(10, 4)),
            grid_kwargs=dict() if grid_ground is None else dict(num_ground=grid_ground),
            grid=grid,
            filepath=filepath,
            train_fraction=train_fraction
        )

    def _get_sampling_functions(self, rng: Rng, num_augmented: Augmented = 7) -> SamplingFunctions:
        return {'payment': (num_augmented, lambda s: rng.choice(np.arange(-2, 9), size=s))}

    def _data_plot(self, figsize: Figsize, tight_layout: TightLayout, **kwargs):
        _, axes = plt.subplots(len(kwargs), 1, sharex='col', sharey='col', figsize=figsize, tight_layout=tight_layout)
        for ax, (title, (x, y)) in zip(axes, kwargs.items()):
            data = pd.concat((x, y), axis=1).astype(int)
            sns.pointplot(data=data, x='payment', y='default', hue='married', scale=0.8, markers=DefaultManager.MARKERS,
                          ci=99, errwidth=1.5, capsize=0.1, dodge=0.25, join=False, ax=ax).set(title=title.capitalize())

    def _augmented_plot(self, aug: pd.DataFrame, figsize: Figsize, tight_layout: TightLayout, **kwargs):
        _, axes = plt.subplots(1, len(self.directions), sharey='all', figsize=figsize, tight_layout=tight_layout)
        for ax, feature in zip(axes, list(self.directions.keys())):
            sns.histplot(data=aug, x=feature, hue='Augmented', discrete=True, ax=ax)
            ticks = np.unique(ax.get_xticks().round().astype(int))
            ax.set_xticks([t for t in ticks if t in range(aug[feature].min(), aug[feature].max() + 1)])

    def _summary_plot(self, model, figsize: Figsize, tight_layout: TightLayout, **kwargs):
        plt.figure(figsize=figsize)
        plt.title('Estimated Function')
        y = model.predict(self.grid).flatten()
        sns.lineplot(data=self.grid, x='payment', y=y, hue='married').set(
            xlabel='Payment Status',
            ylabel='Default Probability'
        )
