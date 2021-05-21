"""Data Manager."""

from typing import List, Callable, Tuple, Dict, Optional as Opt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from moving_targets.util.typing import Monotonicities, Vector, Matrix, Data, Dataset
from src.util.augmentation import get_monotonicities_list, augment_data
from src.util.model import violations_summary, metrics_summary
from src.util.preprocessing import Scaler
from src.util.typing import Augmented, SamplingFunctions, Methods, Figsize, TightLayout, Scalers, AugmentedData, Rng


class DataManager:
    """Abstract dataset handler.
    
    Args:
        x_columns: name of the x features.
        x_scaling: x scaling methods.
        y_column: name of the y feature.
        y_scaling: y scaling method.
        metric: evaluation metric.
        grid: input space grid.
        data_kwargs: plot data arguments.
        augmented_kwargs: plot augmented arguments.
        summary_kwargs: plot summary arguments.
        metric_name: name of the evaluation metric.
        post_process: post-processing routine for the evaluation metric.
    """

    @staticmethod
    def get_kwargs(default: Dict, figsize: Tuple[int, int], tight_layout: bool, **kwargs) -> Dict:
        """Updates the default parameter dictionary.

        Args:
            default: default dictionary.
            figsize: figsize.
            tight_layout: tight_layout.
            **kwargs: additional arguments.

        Returns:
            An updated dictionary.
        """
        output = default.copy()
        if figsize is not None:
            output['figsize'] = figsize
        if tight_layout is not None:
            output[tight_layout] = tight_layout
        output.update(kwargs)
        return output

    def __init__(self,
                 x_columns: List[str],
                 x_scaling: Methods,
                 y_column: str,
                 y_scaling: Methods,
                 metric: Callable,
                 grid: pd.DataFrame,
                 data_kwargs: Dict,
                 augmented_kwargs: Dict,
                 summary_kwargs: Dict,
                 metric_name: str = None,
                 post_process: Callable = None):
        self.x_columns: List[str] = x_columns
        self.x_scaling: Methods = x_scaling
        self.y_column: str = y_column
        self.y_scaling: Methods = y_scaling
        self.metric: Callable = metric
        self.metric_name: str = metric_name
        self.post_process: Callable = post_process
        self.grid: pd.DataFrame = grid
        self.monotonicities: Monotonicities = get_monotonicities_list(
            data=self.grid,
            label=None,
            kind='all',
            errors='ignore',
            compute_monotonicities=self.compute_monotonicities
        )
        self.data_kwargs: Opt = data_kwargs
        self.augmented_kwargs: Opt = augmented_kwargs
        self.summary_kwargs: Opt = summary_kwargs

    def _load_splits(self, **kwargs) -> Dataset:
        raise NotImplementedError("please implement method '_load_splits'")

    def _get_sampling_functions(self, num_augmented: Augmented, rng: Rng) -> SamplingFunctions:
        raise NotImplementedError("please implement method '_get_sampling_functions'")

    def _data_plot(self, **kwargs):
        raise NotImplementedError("please implement method '_data_plot'")

    def _augmented_plot(self, aug: pd.DataFrame, **kwargs):
        figsize, tight_layout = kwargs.get('figsize'), kwargs.get('tight_layout')
        _, axes = plt.subplots(1, len(self.x_columns), sharey='all', figsize=figsize, tight_layout=tight_layout)
        for ax, feature in zip(axes, self.x_columns):
            sns.histplot(data=aug, x=feature, hue='Augmented', ax=ax)

    def _summary_plot(self, model, **kwargs):
        raise NotImplementedError("please implement method '_summary_plot'")

    def compute_monotonicities(self, samples: np.ndarray, references: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Routine to compute the monotonicities."""
        raise NotImplementedError("please implement method '_compute_monotonicities'")

    def get_scalers(self, x: Matrix, y: Vector) -> Tuple[Scaler, Scaler]:
        """Returns the dataset scalers."""
        x_scaler = None if self.x_scaling is None else Scaler(self.x_scaling).fit(x)
        y_scaler = None if self.y_scaling is None else Scaler(self.y_scaling).fit(y)
        return x_scaler, y_scaler

    def load_data(self, **kwargs) -> Tuple[Dataset, Scalers]:
        """Loads the dataset."""
        splits = self._load_splits(**kwargs)
        return splits, self.get_scalers(splits['train'][0], splits['train'][1])

    def get_augmented_data(self,
                           x: Matrix,
                           y: Vector,
                           num_augmented: Augmented = 15,
                           num_random: int = 0,
                           num_ground: Opt[int] = None,
                           seed: int = 0) -> Tuple[AugmentedData, Scalers]:
        """Builds the augmented dataset."""
        rng = np.random.default_rng(seed=seed)
        # handle input samples reduction
        if num_ground is not None:
            x = x.head(num_ground)
            y = y.head(num_ground)
        # add random unsupervised samples to fill the data space
        if num_random > 0:
            random_values = {}
            sampling_functions = self._get_sampling_functions(num_augmented=num_random, rng=rng)
            for col in x.columns:
                # if there is an explicit sampling strategy use it, otherwise sample original data
                n, f = sampling_functions.get(col, (0, lambda s: rng.choice(x[col], size=s)))
                random_values[col] = f(num_random)
            x = pd.concat((x, pd.DataFrame.from_dict(random_values)), ignore_index=True)
            y = pd.concat((y, pd.Series([np.nan] * num_random, name=y.name)), ignore_index=True)
        # augment data
        x_aug, y_aug = augment_data(
            x=x,
            y=y,
            compute_monotonicities=self.compute_monotonicities,
            sampling_functions=self._get_sampling_functions(num_augmented=num_augmented, rng=rng)
        )
        mask = ~np.isnan(y_aug[self.y_column])
        return (x_aug, y_aug), self.get_scalers(x=x_aug, y=y_aug[self.y_column][mask])

    def plot_data(self, figsize: Figsize = None, tight_layout: TightLayout = None, **kwargs: Data):
        """Plots the given data."""
        # print general info about data
        info = [f'{len(x)} {title} samples' for title, (x, _) in kwargs.items()]
        print(', '.join(info))
        # plot data
        kwargs = self.get_kwargs(default=self.data_kwargs, figsize=figsize, tight_layout=tight_layout, **kwargs)
        self._data_plot(**kwargs)
        plt.show()

    def plot_augmented(self, x: Matrix, y: Vector, figsize: Figsize = None, tight_layout: TightLayout = None, **kwargs):
        """Plots the given augmented data."""
        # retrieve augmented data
        aug = x.copy()
        aug['Augmented'] = np.isnan(y[self.y_column])
        # plot augmented data
        kwargs = self.get_kwargs(default=self.augmented_kwargs, figsize=figsize, tight_layout=tight_layout, **kwargs)
        self._augmented_plot(aug=aug, **kwargs)
        plt.show()

    def evaluation_summary(self, model, figsize: Figsize = None, tight_layout: TightLayout = None, **kwargs):
        """Evaluates the model."""
        # compute metrics on kwargs
        print(violations_summary(
            model=model,
            grid=self.grid,
            monotonicities=self.monotonicities
        ))
        print(metrics_summary(
            model=model,
            metric=self.metric,
            metric_name=self.metric_name,
            post_process=self.post_process,
            **kwargs
        ))
        # plot summary
        kwargs = self.get_kwargs(default=self.summary_kwargs, figsize=figsize, tight_layout=tight_layout, **kwargs)
        self._summary_plot(model=model, **kwargs)
        plt.show()
