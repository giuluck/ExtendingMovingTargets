"""Data Manager."""

from typing import List, Callable, Tuple, Dict, Union, Optional

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
    """Abstract dataset manager."""

    Data = Dict[str, pd.DataFrame]
    """Dictionary type that associates to each split name the respective dataframe."""

    DataInfo = Union[Tuple[Dataset, Scalers], List[Tuple[Dataset, Scalers]]]
    """Either a tuple of `Dataset` and `Scalers` or a list of them."""

    Samples = Union[pd.DataFrame, pd.Series]
    """Either a `pandas.DataFrame` or a `pandas.Series`."""

    @staticmethod
    def get_plt_kwargs(default: Dict, figsize: Figsize = None, tight_layout: TightLayout = None, **plt_kwargs) -> Dict:
        """Updates the default parameter dictionary.

        :param default:
            The default dictionary.

        :param figsize:
            `plt.figure()` argument.

        :param tight_layout:
            `plt.figure()` argument.

        :param plt_kwargs:
            Additional `plt.figure()` arguments.

        :returns:
            An updated dictionary, with 'figsize' and 'tight_layout' parameters included and set to None.
        """
        output = default.copy()
        output.update(figsize=figsize, tight_layout=tight_layout, **plt_kwargs)
        return output

    @staticmethod
    def load_data(**data_kwargs) -> Data:
        """Loads the dataset.

        :param data_kwargs:
            Any dataset-dependent argument that may be necessary in the implementation of this method.

        :returns:
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
                 grid_kwargs: Optional[Dict] = None,
                 grid: Optional[pd.DataFrame] = None,
                 loss_name: Optional[str] = None,
                 metric_name: Optional[str] = None,
                 post_process: Callable = None,
                 **load_data_kwargs):
        """
        :param directions:
            Dictionary containing the name of the x features and the respective expected monotonicity.

        :param stratify:
            Whether or not to stratify the dataset when splitting.

        :param x_scaling:
            X scaling methods.

        :param y_scaling:
            Y scaling method.

        :param label:
            Name of the y feature.

        :param loss:
            The evaluation loss.

        :param metric:
            The evaluation metric.

        :param data_kwargs:
            Default `self.plot_data()` arguments.

        :param augmented_kwargs:
            Default `self.plot_augmented()` arguments.

        :param summary_kwargs:
            Default `self.plot_summary()` arguments.

        :param grid_kwargs:
            Default `self.get_augmented_data()` arguments, ignored if an explicit grid is passed.

        :param grid:
            Either an explicit grid for the metric evaluation or None. If None, the grid is obtained by augmenting
            (with the parameters in `grid_kwargs`) the test set which is then used to compute the level of constraint
            satisfaction and the other metrics.

        :param loss_name:
            The name of the evaluation loss.

        :param metric_name:
            The name of the evaluation metric.

        :param post_process:
            Either None (identity function) or an explicit post-processing function f(p) for the predictions, which may
            be used, e.g., to convert the probabilities into output classes for certain metrics.

        :param load_data_kwargs:
            Any dataset-dependent argument to be passed to the static `load_data()` function.
        """
        train, test = self.load_data(**load_data_kwargs).values()
        self.train_data: Tuple[pd.DataFrame, pd.Series] = (train.drop(columns=label), train[label])
        """The training data in the form of a tuple (xtr, ytr)."""

        self.test_data: Tuple[pd.DataFrame, pd.Series] = (test.drop(columns=label), test[label])
        """The test data in the form of a tuple (xts, yts)."""

        self.directions: Dict[str] = directions
        """Dictionary containing the name of the x features and the respective expected monotonicity."""

        self.stratify: Optional[Vector] = self.train_data[1] if stratify else None
        """Whether or not to stratify the dataset when splitting."""

        self.x_scaling: Methods = x_scaling
        """X scaling methods."""

        self.y_scaling: Methods = y_scaling
        """Y scaling method."""

        self.label: str = label
        """Name of the y feature."""

        self.loss: Callable = loss
        """The evaluation loss."""

        self.loss_name: Optional[str] = loss_name
        """The name of the evaluation loss."""

        self.metric: Callable = metric
        """The evaluation metric."""

        self.metric_name: Optional[str] = metric_name
        """The name of the evaluation metric."""

        self.post_process: Callable = post_process
        """Either None (identity function) or an explicit post-processing function f(p) for the predictions, which may
        be used, e.g., to convert the probabilities into output classes for certain metrics."""

        self.data_kwargs: Optional[Dict] = data_kwargs
        """Default `self.plot_data()` arguments."""

        self.augmented_kwargs: Optional[Dict] = augmented_kwargs
        """Default `self.plot_augmented()` arguments."""

        self.summary_kwargs: Optional[Dict] = summary_kwargs
        """Default `self.plot_summary()` arguments."""

        # if an explicit grid is passed we use that, otherwise we build a grid by augmenting the test data, then
        # monotonicities are computed withing groups in case of test data or on all samples in case of explicit grid
        if grid is None:
            aug, _ = self.get_augmented_data(*self.test_data, monotonicities=False, **grid_kwargs)
            grid = pd.concat(aug, axis=1).drop(columns=self.label)
            kind = 'group'
        else:
            kind = 'all'

        self.grid: pd.DataFrame = grid.drop(columns=['ground_index', 'monotonicity'], errors='ignore')
        """The grid for the constraint metrics evaluation."""

        self.monotonicities: MonotonicitiesList = get_monotonicities_list(
            data=grid,
            label=None,
            kind=kind,
            errors='ignore',
            compute_monotonicities=self.compute_monotonicities
        )
        """The list of expected monotonicities related to the grid."""

    def _get_sampling_functions(self, rng: Rng, num_augmented: Augmented) -> SamplingFunctions:
        """Builds the dataset-dependent sampling functions.

        :param rng:
            A random number generator.

        :param num_augmented:
            The number of augmented samples.

        :returns:
            The dictionary of sampling functions.
        """
        raise NotImplementedError("please implement method '_get_sampling_functions'")

    def _data_plot(self, figsize: Figsize, tight_layout: TightLayout, **additional_kwargs):
        """Plotting routine for the original dataset.

        :param figsize:
            The figsize parameter passed to `matplotlib.pyplot.show()`.

        :param tight_layout:
            The tight_layout parameter passed to `matplotlib.pyplot.show()`.

        :param additional_kwargs:
            Any other implementation-dependent parameter.
        """
        raise NotImplementedError("please implement method '_data_plot'")

    def _augmented_plot(self, aug: pd.DataFrame, figsize: Figsize, tight_layout: TightLayout, **additional_kwargs):
        """Plotting routine for the augmented dataset.

        :param aug:
            The augmented DataFrame.

        :param figsize:
            The figsize parameter passed to `matplotlib.pyplot.show()`.

        :param tight_layout:
            The tight_layout parameter passed to `matplotlib.pyplot.show()`.

        :param additional_kwargs:
            Any other implementation-dependent parameter.
        """
        _, axes = plt.subplots(1, len(self.directions), sharey='all', figsize=figsize, tight_layout=tight_layout)
        for ax, feature in zip(axes, list(self.directions.keys())):
            sns.histplot(data=aug, x=feature, hue='Augmented', ax=ax)

    def _summary_plot(self, model, figsize: Figsize, tight_layout: TightLayout, **additional_kwargs):
        """Summary plotting routine.

        :param model:
            A model object having the 'predict(x)' method.

        :param figsize:
            The figsize parameter passed to `matplotlib.pyplot.show()`.

        :param tight_layout:
            The tight_layout parameter passed to `matplotlib.pyplot.show()`.

        :param additional_kwargs:
            Any other implementation-dependent parameter.
        """
        raise NotImplementedError("please implement method '_summary_plot'")

    def compute_monotonicities(self, samples: Samples, references: Samples, eps: float = 1e-5) -> MonotonicitiesMatrix:
        """Routine to compute the monotonicities.

        :param samples:
            The data samples.

        :param references:
            The reference samples.

        :param eps:
            The slack value under which a violation is considered to be acceptable.

        :returns:
            A NxM matrix where N is the number of samples and M is the number of references, where each cell is filled
            with -1, 0, or 1 depending on the kind of monotonicity between samples[i] and references[j].
        """
        directions = np.array([self.directions.get(c) or 0 for c in samples.columns])
        return compute_numeric_monotonicities(samples.values, references.values, directions=directions, eps=eps)

    def get_scalers(self, x: Matrix, y: Vector) -> Tuple[Scaler, Scaler]:
        """Returns the dataset scalers.

        :param x:
            The input data.

        :param y:
            The output data.

        :returns:
            A pair of scalers, one for the input and one for the output data, respectively.
        """
        x_scaler = None if self.x_scaling is None else Scaler(self.x_scaling).fit(x)
        y_scaler = None if self.y_scaling is None else Scaler(self.y_scaling).fit(y)
        return x_scaler, y_scaler

    def get_folds(self, num_folds: Optional[int] = None, **crossval_kwargs) -> DataInfo:
        """Gets the data split in folds.

        With num_folds = None directly returns a tuple with train/test splits and scalers.
        With num_folds = 1 returns a list with a single tuple with train/val/test splits and scalers.
        With num_folds > 1 returns a list of tuples with train/val/test splits and their respective scalers.

        :param num_folds:
            The number of folds for k-fold cross-validation.

        :param crossval_kwargs:
            Arguments passed either to `split_dataset()` or `cross_validate()` method, depending on the number of folds.

        :return:
            Either a tuple of `Dataset` and `Scalers` or a list of them, depending on the number of folds.
        """
        if num_folds is None:
            splits = dict(train=self.train_data, test=self.test_data)
            return splits, self.get_scalers(*self.train_data)
        elif num_folds > 0:
            if num_folds == 1:
                fold = split_dataset(*self.train_data, test_size=0.2, val_size=0.0,
                                     stratify=self.stratify, **crossval_kwargs)
                fold['validation'] = fold.pop('test')
                folds = [fold]
            else:
                folds = cross_validate(*self.train_data, num_folds=num_folds, stratify=self.stratify, **crossval_kwargs)
            return [({**fold, 'test': self.test_data}, self.get_scalers(*fold['train'])) for fold in folds]
        else:
            raise ValueError(f"{num_folds} is not an accepted value for 'num_folds'")

    def get_augmented_data(self,
                           x: pd.DataFrame,
                           y: pd.Series,
                           num_augmented: Optional[Augmented] = None,
                           num_random: int = 0,
                           num_ground: Optional[int] = None,
                           monotonicities: bool = True,
                           seed: int = 0) -> Tuple[AugmentedData, Scalers]:
        """Builds the augmented dataset.

        :param x:
            The input data.

        :param y:
            The output labels.

        :param num_augmented:
            The number of augmented samples.

        :param num_random:
            The number of unlabelled random samples added to the original dataset.

        :param num_ground:
            The number of samples taken from the original dataset (the remaining ones are ignored).

        :param monotonicities:
            Whether or not to compute monotonicities between same-group samples.

        :param seed:
            The random seed.

        :return:
            A tuple containing the augmentation dataset and the respective x/y scalers.
        """
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

    def plot_data(self, **data_kwargs):
        """Plots the given data.

        :param data_kwargs:
            Custom arguments passed to the internal `_data_plot()` method.
        """
        # print general info about data
        info = [f'{len(x)} {title} samples' for title, (x, _) in data_kwargs.items()]
        print(', '.join(info))
        # plot data
        data_kwargs = self.get_plt_kwargs(default=self.data_kwargs, **data_kwargs)
        self._data_plot(**data_kwargs)
        plt.show()

    def plot_augmented(self, x: Matrix, y: Vector, **augmented_kwargs):
        """Plots the given augmented data.

        :param x:
            The augmented input data.

        :param y:
            The augmented output labels.

        :param augmented_kwargs:
            Custom arguments passed to the internal `_augmented_plot()` method.
        """
        # retrieve augmented data
        aug = x.copy()
        aug['Augmented'] = np.isnan(y[self.label])
        # plot augmented data
        augmented_kwargs = self.get_plt_kwargs(default=self.augmented_kwargs, **augmented_kwargs)
        self._augmented_plot(aug=aug, **augmented_kwargs)
        plt.show()

    def losses_summary(self, model, return_type: str = 'str', **kwargs) -> Union[str, Dict[str, float]]:
        """Computes the losses over a custom set of validation data, then builds a summary.

        :param model:
            A model object having the 'predict(x)' method.

        :param return_type:
            Either 'str' to return the string, or 'dict' to return the dictionary.

        :param kwargs:
            A dictionary of named `Data` arguments.

        :returns:
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

        :param model:
            A model object having the 'predict(x)' method.

        :param return_type:
            Either 'str' to return the string, or 'dict' to return the dictionary.

        :param kwargs:
            A dictionary of named `Data` arguments.

        :returns:
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

        :param model:
            A model object having the 'predict(x)' method.

        :param return_type:
            Either 'str' to return the string, or 'dict' to return the dictionary.

        :returns:
            Either a dictionary for the metric values or a string representing the evaluation summary.
        """
        return violations_summary(
            model=model,
            inputs=self.grid,
            monotonicities=self.monotonicities,
            return_type=return_type
        )

    def evaluation_summary(self, model, do_plot: bool = True, model_name: Optional[str] = None, **kwargs):
        """Evaluates the model.

        :param model:
            A model object having the 'predict(x)' method.

        :param do_plot:
            Whether or not to plot the results.

        :param model_name:
            The (optional) name of the model, to be printed in the plot.

        :param kwargs:
            Custom arguments passed to the internal `_summary_plot()` method.
        """
        # compute metrics on kwargs
        print(self.losses_summary(model=model, return_type='str', **kwargs))
        print(self.metrics_summary(model=model, return_type='str', **kwargs))
        print(self.violations_summary(model=model, return_type='str'))
        # plot summary
        kwargs = self.get_plt_kwargs(default=self.summary_kwargs, **kwargs)
        if do_plot:
            self._summary_plot(model=model, **kwargs)
            if model_name:
                plt.suptitle(model_name)
            plt.show()
