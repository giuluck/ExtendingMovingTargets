"""Cars Data Manager."""

from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

from moving_targets.util.typing import Splits
from src.datasets.abstract_manager import AbstractManager
from src.util.preprocessing import split_dataset
from src.util.typing import Augmented, SamplingFunctions, Methods, Rng, Figsize, TightLayout


class CarsManager(AbstractManager):
    """Data Manager for the Cars Dataset."""

    def __init__(self, filepath: str, x_scaling: Methods = 'std', y_scaling: Methods = 'norm',
                 bound: Tuple[int, int] = (0, 100), res: int = 700):
        self.filepath: str = filepath
        self.bound: Tuple[int, int] = bound
        super(CarsManager, self).__init__(
            x_columns=['price'],
            x_scaling=x_scaling,
            y_column='sales',
            y_scaling=y_scaling,
            directions=[-1],
            metric=r2_score,
            metric_name='r2',
            grid=pd.DataFrame.from_dict({'price': np.linspace(self.bound[0], self.bound[1], res)}),
            data_kwargs=dict(figsize=(14, 4), tight_layout=True),
            augmented_kwargs=dict(figsize=(10, 4)),
            summary_kwargs=dict(figsize=(10, 4), res=100, ylim=(-5, 125))
        )

    def _load_splits(self, num_folds: Optional[int], extrapolation: bool) -> Splits:
        # preprocess data
        df = pd.read_csv(self.filepath).rename(columns={'Price in thousands': 'price', 'Sales in thousands': 'sales'})
        df = df[['price', 'sales']].replace({'.': np.nan}).dropna().astype('float')
        # split train/test
        extrapolation = 0.2 if extrapolation else None
        splits = split_dataset(df[['price']], df['sales'], extrapolation=extrapolation, test_size=0.2, val_size=0.0)
        return self.cross_validate(splits, num_folds=num_folds, stratify=False)

    def _get_sampling_functions(self, rng: Rng, num_augmented: Augmented = 15) -> SamplingFunctions:
        return {'price': (num_augmented, lambda s: rng.uniform(self.bound[0], self.bound[1], size=s))}

    def _data_plot(self, figsize: Figsize, tight_layout: TightLayout, **kwargs):
        _, axes = plt.subplots(1, len(kwargs), sharex='all', sharey='all', figsize=figsize, tight_layout=tight_layout)
        for ax, (title, (x, y)) in zip(axes, kwargs.items()):
            sns.scatterplot(x=x['price'], y=y, ax=ax).set(xlabel='price', ylabel='sales', title=title.capitalize())

    def _augmented_plot(self, aug: pd.DataFrame, figsize: Figsize, tight_layout: TightLayout, **kwargs):
        plt.figure(figsize=figsize)
        sns.histplot(data=aug, x='price', hue='Augmented')

    def _summary_plot(self, model, figsize: Figsize, tight_layout: TightLayout, **kwargs):
        res = kwargs.pop('res')
        ylim = kwargs.pop('ylim')
        plt.figure(figsize=figsize)
        for title, (x, y) in kwargs.items():
            sns.scatterplot(x=x['price'], y=y, alpha=0.25, sizes=0.25, label=title.capitalize())
        x = pd.DataFrame(np.linspace(self.bound[0], self.bound[1], res), columns=['price'])
        y = model.predict(x).flatten()
        sns.lineplot(x=x['price'], y=y, color='black').set(ylabel='sales', title='Estimated Function')
        plt.xlim(self.bound)
        plt.ylim(ylim)
