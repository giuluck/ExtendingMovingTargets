"""Synthetic Data Manager."""

from typing import Union, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

from moving_targets.util.typing import Number, Vector
from src.datasets.abstract_manager import AbstractManager
from src.util.plot import ColorFader
from src.util.preprocessing import split_dataset
from src.util.typing import Augmented, Rng, SamplingFunctions, Figsize, TightLayout


class SyntheticManager(AbstractManager):
    """Data Manager for the Synthetic Dataset."""

    @staticmethod
    def function(a: Union[Vector, Number], b: Union[Vector, Number]) -> Union[Vector, Number]:
        """Ground function.

        :param a:
            Either a single floating point value or a vector of floating point values representing the first feature.

        :param b:
            Either a single floating point value or a vector of floating point values representing the second feature.

        :returns:
            Either a single floating point value or a vector of floating point values representing the function output.
        """
        a = a ** 3
        b = np.sin(np.pi * (b - 0.01)) ** 2 + 1
        return a / b + b

    @staticmethod
    def sample_dataset(n: int, noise: float, rng: Rng, testing_set: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Sample data points with the given amount of noise.

        :param n:
            The number of samples.

        :param noise:
            The amount of output noise when sampling the function.

        :param rng:
            A random number generator.

        :param testing_set:
            Whether or not to use test set sampling distributions when sampling the input features.

        :returns:
            A tuple (x, y) containing the input data and the output labels.
        """
        a = rng.uniform(low=-1, high=1, size=n) if testing_set else rng.normal(scale=0.3, size=n).clip(min=-1, max=1)
        b = rng.uniform(low=-1, high=1, size=n)
        x = pd.DataFrame.from_dict({'a': a, 'b': b})
        y = pd.Series(SyntheticManager.function(a, b), name='label') + rng.normal(scale=noise, size=len(x))
        return x, y

    @staticmethod
    def load_data(noise: float = 0.0, extrapolation: bool = False) -> AbstractManager.Data:
        """Loads the dataset.

        :param noise:
            The amount of output noise to add to the labels.

        :param extrapolation:
            Whether to consider a test set with border samples or with random samples (interpolation).

        :returns:
            A dictionary of dataframes representing the train and test sets, respectively.
        """
        rng = np.random.default_rng(seed=0)
        if extrapolation:
            x, y = SyntheticManager.sample_dataset(n=700, noise=noise, rng=rng, testing_set=True)
            splits = split_dataset(x, y, extrapolation={'a': 0.7}, val_size=0.25, random_state=0)
        else:
            splits = {
                'train': SyntheticManager.sample_dataset(n=200, noise=noise, rng=rng, testing_set=False),
                'test': SyntheticManager.sample_dataset(n=500, noise=noise, rng=rng, testing_set=True)
            }
        return {split: pd.concat(data, axis=1) for split, data in splits.items()}

    def __init__(self, noise: float = 0.0, extrapolation: bool = False, full_grid: bool = False, x_scaling: str = 'std',
                 y_scaling: str = 'norm', grid_augmented: int = 35, grid_ground: Optional[int] = None):
        """
        :param noise:
            The amount of output noise to add to the labels.

        :param extrapolation:
            Whether or not to test on extrapolation instead of interpolation.

        :param full_grid:
            Whether or not to evaluate the results on an explicit full grid. This option is possible only if
            full_features is set to false, since there is no way to create an explicit grid on the full set of features.

        :param x_scaling:
            Scaling methods for the input data.

        :param y_scaling:
            Scaling methods for the output data.

        :param grid_augmented:
            Number of augmented features for grid_kwargs. This will not have any effect if an explicit grid is passed.

        :param grid_ground:
            Number of ground samples for grid_kwargs. This will not have any effect if an explicit grid is passed.
        """
        if full_grid:
            a, b = np.meshgrid(np.linspace(-1, 1, 80), np.linspace(-1, 1, 80))
            grid = pd.DataFrame({'a': a.flatten(), 'b': b.flatten()})
        else:
            grid = None
        super(SyntheticManager, self).__init__(
            directions={'a': 1, 'b': 0},
            stratify=False,
            x_scaling=x_scaling,
            y_scaling=y_scaling,
            label='label',
            loss=mean_squared_error,
            loss_name='mse',
            metric=r2_score,
            metric_name='r2',
            data_kwargs=dict(figsize=(12, 10), tight_layout=True),
            augmented_kwargs=dict(figsize=(10, 4), tight_layout=True),
            summary_kwargs=dict(figsize=(10, 4), tight_layout=True, res=50),
            grid_kwargs=dict(num_augmented=grid_augmented, num_ground=grid_ground),
            grid=grid,
            noise=noise,
            extrapolation=extrapolation
        )

    def _get_sampling_functions(self, rng: Rng, num_augmented: Augmented = 15) -> SamplingFunctions:
        return {'a': (num_augmented, lambda s: rng.uniform(-1, 1, size=s))}

    def _data_plot(self, figsize: Figsize, tight_layout: TightLayout, **additional_kwargs):
        _, ax = plt.subplots(3, len(additional_kwargs), sharex='row', sharey='row', figsize=figsize, tight_layout=tight_layout)
        # hue/size bounds
        ybn = np.concatenate([[y.min(), y.max()] for _, y in additional_kwargs.values()])
        abn, bbn, ybn = (-1, 1), (-1, 1), (np.min(ybn), np.max(ybn))
        # plots
        for i, (title, (x, y)) in enumerate(additional_kwargs.items()):
            a, b = x['a'], x['b']
            ax[0, i].set(title=title.capitalize())
            sns.scatterplot(x=a, y=y, hue=b, hue_norm=bbn, size=b, size_norm=bbn, ax=ax[0, i], legend=False)
            ax[0, i].legend([f'b {bbn}'], markerscale=0, handlelength=0)
            sns.scatterplot(x=b, y=y, hue=a, hue_norm=abn, size=a, size_norm=abn, ax=ax[1, i], legend=False)
            ax[1, i].legend([f'a {abn}'], markerscale=0, handlelength=0)
            sns.scatterplot(x=a, y=b, hue=y, hue_norm=ybn, size=y, size_norm=ybn, ax=ax[2, i], legend=False)
            ax[2, i].legend([f'label ({ybn[0]:.0f}, {ybn[1]:.0f})'], markerscale=0, handlelength=0)

    def _summary_plot(self, model, figsize: Figsize, tight_layout: TightLayout, **additional_kwargs):
        res = additional_kwargs.pop('res')
        # get data
        a, b = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
        grid = pd.DataFrame.from_dict({'a': a.flatten(), 'b': b.flatten()})
        grid['pred'] = model.predict(grid)
        grid['label'] = SyntheticManager.function(grid['a'], grid['b'])
        fader = ColorFader('red', 'blue', bounds=[-1, 1])
        _, axes = plt.subplots(2, 3, figsize=figsize, tight_layout=tight_layout)
        for ax, (title, y) in zip(axes, {'Ground Truth': 'label', 'Estimated Function': 'pred'}.items()):
            # plot bivariate function
            z = grid[y].values.reshape(res, res)
            ax[0].pcolor(a, b, z, shading='auto', cmap='viridis', vmin=grid['label'].min(), vmax=grid['label'].max())
            # plot first feature (with title as it is the central plot)
            for idx, group in grid.groupby('b'):
                label = f'b = {idx:.0f}' if idx in [-1, 1] else None
                sns.lineplot(data=group, x='a', y=y, color=fader(idx), alpha=0.4, label=label, ax=ax[1])
            # plot second feature
            for idx, group in grid.groupby('a'):
                label = f'a = {idx:.0f}' if idx in [-1, 1] else None
                sns.lineplot(data=group, x='b', y=y, color=fader(idx), alpha=0.4, label=label, ax=ax[2])
