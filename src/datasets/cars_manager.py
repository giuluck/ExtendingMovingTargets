"""Cars Data Manager."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Tuple, List
from sklearn.metrics import r2_score

from moving_targets.util.typing import Dataset
from src.datasets.data_manager import DataManager
from src.util.augmentation import compute_numeric_monotonicities
from src.util.preprocessing import split_dataset, cross_validate
from src.util.typing import Augmented, SamplingFunctions, Methods, Rng, Figsize, TightLayout


class CarsManager(DataManager):
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
            metric=r2_score,
            metric_name='r2',
            grid=pd.DataFrame.from_dict({'price': np.linspace(self.bound[0], self.bound[1], res)}),
            data_kwargs=dict(figsize=(14, 4), tight_layout=True),
            augmented_kwargs=dict(figsize=(10, 4)),
            summary_kwargs=dict(figsize=(10, 4), res=100, ylim=(-5, 125))
        )

    # noinspection PyMissingOrEmptyDocstring
    def compute_monotonicities(self, samples: np.ndarray, references: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        return compute_numeric_monotonicities(samples, references, directions=[-1], eps=eps)

    def _load_splits(self, num_folds: int, extrapolation: bool) -> List[Dataset]:
        # preprocess data
        df = pd.read_csv(self.filepath).rename(columns={'Price in thousands': 'price', 'Sales in thousands': 'sales'})
        df = df[['price', 'sales']].replace({'.': np.nan}).dropna().astype('float')
        x, y = df[['price']], df['sales']
        # split data
        if num_folds == 1:
            extrapolation = 0.2 if extrapolation else None
            fold = split_dataset(x, y, extrapolation=extrapolation, test_size=0.2, val_size=0.2, random_state=0)
            return [fold]
        else:
            return cross_validate(x, y, num_folds=num_folds, shuffle=True, random_state=0)

    def _get_sampling_functions(self, num_augmented: Augmented, rng: Rng) -> SamplingFunctions:
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
        x = np.linspace(self.bound[0], self.bound[1], res)
        y = model.predict(x.reshape(-1, 1)).flatten()
        sns.lineplot(x=x, y=y, color='black').set(xlabel='price', ylabel='sales', title='Estimated Function')
        plt.xlim(self.bound)
        plt.ylim(ylim)
