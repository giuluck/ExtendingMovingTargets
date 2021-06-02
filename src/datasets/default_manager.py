"""Default Data Manager."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score

from moving_targets.util.typing import Splits
from src.datasets.data_manager import DataManager
from src.util.preprocessing import split_dataset
from src.util.typing import Augmented, SamplingFunctions, Rng, Figsize, TightLayout


class DefaultManager(DataManager):
    """Data Manager for the Default Dataset."""

    MARKERS = {k: v for k, v in enumerate(['o', 's', '^', '+'])}

    def __init__(self, filepath: str, x_scaling: str = 'std', y_scaling: str = 'norm', train_fraction: float = 0.025):
        self.filepath: str = filepath
        self.train_fraction: float = train_fraction
        married, payment = np.meshgrid([0, 1], np.arange(-2, 9))
        super(DefaultManager, self).__init__(
            x_columns=['married', 'payment'],
            x_scaling={'payment': x_scaling},
            y_column='default',
            y_scaling=y_scaling,
            directions=[0, 1],
            metric=accuracy_score,
            metric_name='acc',
            post_process=lambda x: x.round().astype(int),
            grid=pd.DataFrame.from_dict({'married': married.flatten(), 'payment': payment.flatten()}),
            data_kwargs=dict(figsize=(12, 10), tight_layout=True),
            augmented_kwargs=dict(figsize=(10, 4), tight_layout=True),
            summary_kwargs=dict(figsize=(10, 4))
        )

    def _load_splits(self, num_folds: Optional[int], extrapolation: bool) -> Splits:
        assert extrapolation is False, "'extrapolation' is not supported for Default dataset"
        # preprocess data
        df = pd.read_csv(self.filepath)[['MARRIAGE', 'PAY_0', 'default']]
        df = df.dropna().reset_index(drop=True)
        df = df[np.in1d(df['MARRIAGE'], [1, 2])]
        df['MARRIAGE'] = df['MARRIAGE'] - 1
        df = df.rename(columns={'MARRIAGE': 'married', 'PAY_0': 'payment'})
        df = df.astype({'married': float, 'payment': float, 'default': int})
        x, y = df[['married', 'payment']], df['default']
        # split train/test
        splits = split_dataset(x, y, test_size=1 - self.train_fraction, val_size=0.0, stratify=y)
        return self.cross_validate(splits=splits, num_folds=num_folds, stratify=True)

    def _get_sampling_functions(self, rng: Rng, num_augmented: Augmented = 7) -> SamplingFunctions:
        return {'payment': (num_augmented, lambda s: rng.choice(np.arange(-2, 9), size=s))}

    def _data_plot(self, figsize: Figsize, tight_layout: TightLayout, **kwargs):
        _, axes = plt.subplots(len(kwargs), 1, sharex='col', sharey='col', figsize=figsize, tight_layout=tight_layout)
        for ax, (title, (x, y)) in zip(axes, kwargs.items()):
            data = pd.concat((x, y), axis=1).astype(int)
            sns.pointplot(data=data, x='payment', y='default', hue='married', scale=0.8, markers=DefaultManager.MARKERS,
                          ci=99, errwidth=1.5, capsize=0.1, dodge=0.25, join=False, ax=ax).set(title=title.capitalize())

    def _augmented_plot(self, aug: pd.DataFrame, figsize: Figsize, tight_layout: TightLayout, **kwargs):
        _, axes = plt.subplots(1, len(self.x_columns), sharey='all', figsize=figsize, tight_layout=tight_layout)
        for ax, feature in zip(axes, self.x_columns):
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
