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

    # https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
    FEATURES: Dict[str, FeatureInfo] = {
        'default': FeatureInfo(kind='float', alias=None),
        'LIMIT_BAL': FeatureInfo(kind='float', alias='limit'),
        'SEX': FeatureInfo(kind='category', alias='sex'),
        'EDUCATION': FeatureInfo(kind='category', alias='education'),
        'MARRIAGE': FeatureInfo(kind='category', alias='marriage'),
        'AGE': FeatureInfo(kind='float', alias='age'),
        'PAY_0': FeatureInfo(kind='float', alias='sep_status'),
        'PAY_2': FeatureInfo(kind='float', alias='aug_status'),
        'PAY_3': FeatureInfo(kind='float', alias='jul_status'),
        'PAY_4': FeatureInfo(kind='float', alias='jun_status'),
        'PAY_5': FeatureInfo(kind='float', alias='may_status'),
        'PAY_6': FeatureInfo(kind='float', alias='apr_status'),
        'BILL_AMT1': FeatureInfo(kind='float', alias='sep_bill'),
        'BILL_AMT2': FeatureInfo(kind='float', alias='aug_bill'),
        'BILL_AMT3': FeatureInfo(kind='float', alias='jul_bill'),
        'BILL_AMT4': FeatureInfo(kind='float', alias='jun_bill'),
        'BILL_AMT5': FeatureInfo(kind='float', alias='may_bill'),
        'BILL_AMT6': FeatureInfo(kind='float', alias='apr_bill'),
        'PAY_AMT1': FeatureInfo(kind='float', alias='sep_payment'),
        'PAY_AMT2': FeatureInfo(kind='float', alias='aug_payment'),
        'PAY_AMT3': FeatureInfo(kind='float', alias='jul_payment'),
        'PAY_AMT4': FeatureInfo(kind='float', alias='jun_payment'),
        'PAY_AMT5': FeatureInfo(kind='float', alias='may_payment'),
        'PAY_AMT6': FeatureInfo(kind='float', alias='apr_payment')
    }
    MARKERS = {k: v for k, v in enumerate(['o', 's', '^', '+'])}

    # noinspection PyMissingOrEmptyDocstring
    @staticmethod
    def load_data(filepath: str, full_features: bool, train_fraction: float) -> AbstractManager.Data:
        df = pd.read_csv(filepath)
        df = clean_dataframe(df, DefaultManager.FEATURES)
        if full_features:
            df = pd.get_dummies(df.dropna(), prefix_sep=': ')
        else:
            df['marriage'] = df['marriage'].astype('float') - 1
            df = df[np.in1d(df['marriage'], [0, 1])]
            df = df[['marriage', 'sep_status', 'default']].dropna()
        return split_dataset(df, test_size=1 - train_fraction, val_size=0.0, stratify=df['default'])

    def __init__(self, filepath: str, full_features: bool = False, full_grid: bool = False, grid_augmented: int = 10,
                 grid_ground: Optional[int] = None, train_fraction: float = 0.8, x_scaling: str = 'std'):
        grid = None
        if full_features:
            assert full_grid is False, "'full_grid' is not supported with 'full_features'"
            self.months = ['sep', 'aug', 'jul', 'jun', 'may', 'apr']
            x_scaling = {v.alias or k: x_scaling for k, v in DefaultManager.FEATURES.items() if v.kind == 'float'}
        else:
            self.months = ['sep']
            x_scaling = {'sep_status': x_scaling}
            if full_grid:
                marriage, status = np.meshgrid([0, 1], np.arange(-2, 9))
                grid = pd.DataFrame.from_dict({'marriage': marriage.flatten(), 'sep_status': status.flatten()})
        super(DefaultManager, self).__init__(
            directions={f'{c}_status': 1 for c in self.months},
            stratify=True,
            x_scaling=x_scaling,
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
            grid_kwargs=dict(num_augmented=grid_augmented, num_ground=grid_ground),
            grid=grid,
            filepath=filepath,
            full_features=full_features,
            train_fraction=train_fraction
        )

    def _get_sampling_functions(self, rng: Rng, num_augmented: Augmented = 7) -> SamplingFunctions:
        num_augmented = np.round(num_augmented / np.sqrt(len(self.months))).astype(int)
        return {f'{m}_status': (num_augmented, lambda s: rng.choice(np.arange(-2, 9), size=s)) for m in self.months}

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
