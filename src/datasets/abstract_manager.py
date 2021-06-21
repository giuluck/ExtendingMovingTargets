"""Data Manager."""

from typing import List, Callable, Tuple, Dict, Union, Optional as Opt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from moving_targets.util.typing import Vector, Matrix, Dataset, MonotonicitiesMatrix, MonotonicitiesList
from src.util.augmentation import augment_data, compute_numeric_monotonicities, get_monotonicities_list
from src.util.model import violations_summary, metrics_summary
from src.util.preprocessing import Scaler, Scalers, split_dataset, cross_validate
from src.util.typing import Augmented, SamplingFunctions, Methods, Figsize, TightLayout, AugmentedData, Rng


class AbstractManager:
    """Abstract dataset handler.

    Args:
        x_scaling: x scaling methods.
        y_scaling: y scaling method.
        directions: dictionary containing the name of the x features and the respective expected monotonicity.
        label: name of the y feature.
        loss: evaluation loss.
        metric: evaluation metric.
        data_kwargs: plot data arguments.
        augmented_kwargs: plot augmented arguments.
        summary_kwargs: plot summary arguments.
        grid_kwargs: grid augmentation arguments, ignored if is not None.
        grid: either an explicit grid or None.
        loss_name: name of the evaluation loss.
        metric_name: name of the evaluation metric.
        post_process: post-processing routine for the evaluation metric.
        **kwargs: any additional argument to be passed to the `load_data()` function.
    """

    Data = Dict[str, pd.DataFrame]
    DataInfo = Union[Tuple[Dataset, Scalers], List[Tuple[Dataset, Scalers]]]
    Samples = Union[pd.DataFrame, pd.Series]

    @staticmethod
    def get_kwargs(default: Dict, **kwargs) -> Dict:
        """Updates the default parameter dictionary.

        Args:
            default: the default dictionary.
            **kwargs: additional arguments.

        Returns:
            An updated dictionary, with 'figsize' and 'tight_layout' parameters included and set to None.
        """
        output = dict(figsize=None, tight_layout=None)
        output.update(default)
        output.update(kwargs)
        return output

    @staticmethod
    def load_data(**kwargs) -> Data:
        """Loads the dataset.

        Args:
            **kwargs: any argument used in the function.

        Returns:
            A dictionary of dataframes representing the train and test sets, respectively.
        """
        raise NotImplementedError("please implement static method 'load_data()'")

    def __init__(self,
                 directions: Dict[str, int],
                 stratify: bool,
                 x_scaling: Methods,
                 y_scaling: Methods,
                 label: str,
                 loss: Callable,
                 metric: Callable,
                 data_kwargs: Dict,
                 augmented_kwargs: Dict,
                 summary_kwargs: Dict,
                 grid_kwargs: Opt[Dict] = None,
                 grid: Opt[pd.DataFrame] = None,
                 loss_name: Opt[str] = None,
                 metric_name: Opt[str] = None,
                 post_process: Callable = None,
                 **kwargs):
        train, test = self.load_data(**kwargs).values()
        self.train_data: Tuple[pd.DataFrame, pd.Series] = (train.drop(columns=label), train[label])
        self.test_data: Tuple[pd.DataFrame, pd.Series] = (test.drop(columns=label), test[label])
        self.directions: Dict[str] = directions
        self.stratify: Opt[Vector] = self.train_data[1] if stratify else None
        self.x_scaling: Methods = x_scaling
        self.y_scaling: Methods = y_scaling
        self.label: str = label
        self.loss: Callable = loss
        self.loss_name: Opt[str] = loss_name
        self.metric: Callable = metric
        self.metric_name: Opt[str] = metric_name
        self.post_process: Callable = post_process
        self.data_kwargs: Opt[Dict] = data_kwargs
        self.augmented_kwargs: Opt[Dict] = augmented_kwargs
        self.summary_kwargs: Opt[Dict] = summary_kwargs
        # if an explicit grid is passed we use that, otherwise we build a grid by augmenting the test data, then
        # monotonicities are computed withing groups in case of test data or on all samples in case of explicit grid
        if grid is None:
            aug, _ = self.get_augmented_data(*self.test_data, monotonicities=False, **grid_kwargs)
            grid = pd.concat(aug, axis=1).drop(columns=self.label)
            kind = 'group'
        else:
            kind = 'all'
        self.grid: pd.DataFrame = grid.drop(columns=['ground_index', 'monotonicity'], errors='ignore')
        self.monotonicities: MonotonicitiesList = get_monotonicities_list(
            data=grid,
            label=None,
            kind=kind,
            errors='ignore',
            compute_monotonicities=self.compute_monotonicities
        )

    def _get_sampling_functions(self, rng: Rng, num_augmented: Augmented) -> SamplingFunctions:
        raise NotImplementedError("please implement method '_get_sampling_functions'")

    def _data_plot(self, figsize: Figsize, tight_layout: TightLayout, **kwargs):
        raise NotImplementedError("please implement method '_data_plot'")

    def _augmented_plot(self, aug: pd.DataFrame, figsize: Figsize, tight_layout: TightLayout, **kwargs):
        _, axes = plt.subplots(1, len(self.directions), sharey='all', figsize=figsize, tight_layout=tight_layout)
        for ax, feature in zip(axes, list(self.directions.keys())):
            sns.histplot(data=aug, x=feature, hue='Augmented', ax=ax)

    def _summary_plot(self, model, figsize: Figsize, tight_layout: TightLayout, **kwargs):
        raise NotImplementedError("please implement method '_summary_plot'")

    def compute_monotonicities(self, samples: Samples, references: Samples, eps: float = 1e-5) -> MonotonicitiesMatrix:
        """Routine to compute the monotonicities."""
        directions = np.array([self.directions.get(c) or 0 for c in samples.columns])
        return compute_numeric_monotonicities(samples.values, references.values, directions=directions, eps=eps)

    def get_scalers(self, x: Matrix, y: Vector) -> Tuple[Scaler, Scaler]:
        """Returns the dataset scalers."""
        x_scaler = None if self.x_scaling is None else Scaler(self.x_scaling).fit(x)
        y_scaler = None if self.y_scaling is None else Scaler(self.y_scaling).fit(y)
        return x_scaler, y_scaler

    def get_folds(self, num_folds: Opt[int] = None, **kwargs) -> DataInfo:
        """Gets the data split in folds.

        With num_folds = None directly returns a tuple with train/test splits and scalers.
        With num_folds = 1 returns a list with a single tuple with train/val/test splits and scalers.
        With num_folds > 1 returns a list of tuples with train/val/test splits and their respective scalers.
        """
        if num_folds is None:
            splits = dict(train=self.train_data, test=self.test_data)
            return splits, self.get_scalers(*self.train_data)
        elif num_folds > 0:
            if num_folds == 1:
                fold = split_dataset(*self.train_data, test_size=0.2, val_size=0.0, stratify=self.stratify, **kwargs)
                fold['validation'] = fold.pop('test')
                folds = [fold]
            else:
                folds = cross_validate(*self.train_data, num_folds=num_folds, stratify=self.stratify, **kwargs)
            return [({**fold, 'test': self.test_data}, self.get_scalers(*fold['train'])) for fold in folds]
        else:
            raise ValueError(f"{num_folds} is not an accepted value for 'num_folds'")

    def get_augmented_data(self,
                           x: pd.DataFrame,
                           y: pd.Series,
                           num_augmented: Opt[Augmented] = None,
                           num_random: int = 0,
                           num_ground: Opt[int] = None,
                           monotonicities: bool = True,
                           seed: int = 0) -> Tuple[AugmentedData, Scalers]:
        """Builds the augmented dataset."""
        rng = np.random.default_rng(seed=seed)
        x, y = x.reset_index(drop=True), y.reset_index(drop=True)
        # handle input samples reduction
        if num_ground is not None:
            x = x.head(num_ground)
            y = y.head(num_ground)
        # add random unsupervised samples to fill the data space
        if num_random > 0:
            random_values = {}
            sampling_functions = self._get_sampling_functions(rng=rng, num_augmented=num_random)
            for col in x.columns:
                # if there is an explicit sampling strategy use it, otherwise sample original data
                n, f = sampling_functions.get(col, (0, lambda s: rng.choice(x[col], size=s)))
                random_values[col] = f(num_random)
            x = pd.concat((x, pd.DataFrame.from_dict(random_values)), ignore_index=True)
            y = pd.concat((y, pd.Series([np.nan] * num_random, name=y.name)), ignore_index=True)
        # augment data
        sampling_args = {} if num_augmented is None else {'num_augmented': num_augmented}  # if None, uses default
        x_aug, y_aug = augment_data(x=x,
                                    y=y,
                                    sampling_functions=self._get_sampling_functions(rng=rng, **sampling_args),
                                    compute_monotonicities=self.compute_monotonicities if monotonicities else None)
        mask = ~np.isnan(y_aug[self.label])
        return (x_aug, y_aug), self.get_scalers(x=x_aug, y=y_aug[self.label][mask])

    def plot_data(self, **kwargs):
        """Plots the given data."""
        # print general info about data
        info = [f'{len(x)} {title} samples' for title, (x, _) in kwargs.items()]
        print(', '.join(info))
        # plot data
        kwargs = self.get_kwargs(default=self.data_kwargs, **kwargs)
        self._data_plot(**kwargs)
        plt.show()

    def plot_augmented(self, x: Matrix, y: Vector, **kwargs):
        """Plots the given augmented data."""
        # retrieve augmented data
        aug = x.copy()
        aug['Augmented'] = np.isnan(y[self.label])
        # plot augmented data
        kwargs = self.get_kwargs(default=self.augmented_kwargs, **kwargs)
        self._augmented_plot(aug=aug, **kwargs)
        plt.show()

    def losses_summary(self, model, return_type: str = 'str', **kwargs) -> Union[str, Dict[str, float]]:
        """Computes the losses over a custom set of validation data, then builds a summary.

        Args:
            model: a model object having the 'predict(x)' method.
            return_type: either 'str' to return the string, or 'dict' to return the dictionary.
            **kwargs: a dictionary of named `Data` arguments.

        Returns:
            Either a dictionary for the metric values or a string representing the evaluation summary.
        """
        return metrics_summary(
            model=model,
            metric=self.loss,
            metric_name=self.loss_name,
            post_process=None,
            return_type=return_type,
            **kwargs
        )

    def metrics_summary(self, model, return_type: str = 'str', **kwargs) -> Union[str, Dict[str, float]]:
        """Computes the metrics over a custom set of validation data, then builds a summary.

        Args:
            model: a model object having the 'predict(x)' method.
            return_type: either 'str' to return the string, or 'dict' to return the dictionary.
            **kwargs: a dictionary of named `Data` arguments.

        Returns:
            Either a dictionary for the metric values or a string representing the evaluation summary.
        """
        return metrics_summary(
            model=model,
            metric=self.metric,
            metric_name=self.metric_name,
            post_process=self.post_process,
            return_type=return_type,
            **kwargs
        )

    def violations_summary(self, model, return_type: str = 'str') -> Union[str, Dict[str, float]]:
        """Computes the violations over a custom set of validation data, then builds a summary.

        Args:
            model: a model object having the 'predict(x)' method.
            return_type: either 'str' to return the string, or 'dict' to return the dictionary.

        Returns:
            Either a dictionary for the metric values or a string representing the evaluation summary.
        """
        return violations_summary(
            model=model,
            inputs=self.grid,
            monotonicities=self.monotonicities,
            return_type=return_type
        )

    def evaluation_summary(self, model, do_plot: bool = True, model_name: Opt[str] = None, **kwargs):
        """Evaluates the model."""
        # compute metrics on kwargs
        print(self.losses_summary(model=model, return_type='str', **kwargs))
        print(self.metrics_summary(model=model, return_type='str', **kwargs))
        print(self.violations_summary(model=model, return_type='str'))
        # plot summary
        kwargs = self.get_kwargs(default=self.summary_kwargs, **kwargs)
        if do_plot:
            self._summary_plot(model=model, **kwargs)
            if model_name:
                plt.suptitle(model_name)
            plt.show()
