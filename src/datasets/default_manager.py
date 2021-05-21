from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from src.datasets.data_manager import DataManager
from src.util.preprocessing import split_dataset
from src.util.augmentation import compute_numeric_monotonicities


class DefaultManager(DataManager):
    MARKERS = {k: v for k, v in enumerate(['o', 's', '^', '+'])}

    def __init__(self, filepath: str, x_scaling: Any = 'std', y_scaling: Any = 'norm', test_size: float = 0.8):
        self.filepath: str = filepath
        self.test_size: float = test_size
        married, payment = np.meshgrid([0, 1], np.arange(-2, 9))
        super(DefaultManager, self).__init__(
            x_columns=['married', 'payment'],
            x_scaling={'payment': x_scaling},
            y_column='default',
            y_scaling=y_scaling,
            metric=accuracy_score,
            metric_name='acc',
            post_process=lambda x: x.round().astype(int),
            grid=pd.DataFrame.from_dict({'married': married.flatten(), 'payment': payment.flatten()}),
            data_kwargs=dict(figsize=(12, 10), tight_layout=True),
            augmented_kwargs=dict(figsize=(10, 4), tight_layout=True),
            summary_kwargs=dict(figsize=(10, 4))
        )

    def compute_monotonicities(self, samples, references, eps=1e-5):
        return compute_numeric_monotonicities(samples, references, directions=[0, 1], eps=eps)

    def _load_splits(self):
        # preprocess data
        df = pd.read_csv(self.filepath)[['MARRIAGE', 'PAY_0', 'default']]
        df = df.dropna().reset_index(drop=True)
        df = df[np.in1d(df['MARRIAGE'], [1, 2])]
        df['MARRIAGE'] = df['MARRIAGE'] - 1
        df = df.rename(columns={'MARRIAGE': 'married', 'PAY_0': 'payment'}).astype(float)
        # split data
        x, y = df[['married', 'payment']], df['default']
        return split_dataset(x, y, test_size=self.test_size, val_size=0.5, random_state=0)

    def _get_sampling_functions(self, num_augmented, rng):
        return {'payment': (num_augmented, lambda s: rng.choice(np.arange(-2, 9), size=s))}

    def _data_plot(self, figsize, tight_layout, **kwargs):
        _, axes = plt.subplots(len(kwargs), 1, sharex='col', sharey='col', figsize=figsize, tight_layout=tight_layout)
        for ax, (title, (x, y)) in zip(axes, kwargs.items()):
            data = pd.concat((x, y), axis=1).astype(int)
            sns.pointplot(data=data, x='payment', y='default', hue='married', scale=0.8, markers=DefaultManager.MARKERS,
                          ci=99, errwidth=1.5, capsize=0.1, dodge=0.25, join=False, ax=ax).set(title=title.capitalize())

    # noinspection PyMethodOverriding
    def _augmented_plot(self, aug, figsize, tight_layout, **kwargs):
        _, axes = plt.subplots(1, len(self.x_columns), sharey='all', figsize=figsize, tight_layout=tight_layout)
        for ax, feature in zip(axes, self.x_columns):
            sns.histplot(data=aug, x=feature, hue='Augmented', discrete=True, ax=ax)
            ticks = np.unique(ax.get_xticks().round().astype(int))
            ax.set_xticks([t for t in ticks if t in range(aug[feature].min(), aug[feature].max() + 1)])

    # noinspection PyMethodOverriding
    def _summary_plot(self, model, figsize, **kwargs):
        plt.figure(figsize=figsize)
        plt.title('Estimated Function')
        y = model.predict(self.grid).flatten()
        sns.lineplot(data=self.grid, x='payment', y=y, hue='married').set(
            xlabel='Payment Status',
            ylabel='Default Probability'
        )
