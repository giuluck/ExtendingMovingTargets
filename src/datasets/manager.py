"""Data Manager."""
from typing import List, Any, Optional, Tuple, Dict, Callable
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from moving_targets.callbacks import Callback
from moving_targets.metrics import CrossEntropy, Accuracy, Metric, MSE, R2
from moving_targets.util.errors import not_implemented_message
from moving_targets.util.scalers import Scaler

from src.util.analysis import set_pandas_options
from src.util.metrics import GridConstraint
from src.util.preprocessing import split_dataset, cross_validate


class Fold:
    """Data class containing the information of a fold for k-fold cross-validation.

    - 'data' is the training data.
    - 'label' is the target column name.
    - 'validation' is the  shared validation dataset which is common among all the k folds.
    """

    def __init__(self, data: pd.DataFrame, label: str, validation: Dict[str, pd.DataFrame]):
        def split_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
            return df.drop(columns=label), df[label].values

        self.x, self.y = split_df(df=data)
        self.validation: Dict[str, Tuple[pd.DataFrame, np.ndarray]] = {k: split_df(df=v) for k, v in validation.items()}


class Manager:
    """Abstract Dataset Manager.

    - 'label' is the name of the target feature.
    - 'directions' is a dictionary pairing each monotonic attribute to its direction (-1, 0, or 1).
    - 'grid' is the explicit evaluation grid for the dataset.
    - 'classification' is a boolean value representing whether the dataset implies a classification task or not.
    - 'metrics' is the list of metrics that must be evaluated.
    """

    __name__: str = 'dataset'

    callbacks: Dict[str, Callable] = {
        'distance': lambda fs: DistanceAnalysis(file_signature=fs)
    }
    """Callback aliases paired with callable function of type f(<file_signature>) -> <callback>."""

    @classmethod
    def load(cls) -> Dict[str, pd.DataFrame]:
        """Loads the dataset and returns a dictionary of dataframes representing the train and test sets."""
        raise NotImplementedError(not_implemented_message(name='data', static=True))

    @classmethod
    def grid(cls, plot: bool = True) -> pd.DataFrame:
        """Builds the explicit grid.

        - 'plot' is True if the grid is used for plotting, False if it is used for monotonicity evaluation.
        """
        raise NotImplementedError(not_implemented_message(name='grid', static=True))

    def __init__(self,
                 label: str,
                 classification: bool,
                 directions: Dict[str, int]):
        train, test = self.load().values()
        grid = self.grid(plot=False)

        self.train: pd.DataFrame = train
        self.test: pd.DataFrame = test
        self.label: str = label
        self.classification: bool = classification
        self.directions: Dict[str, int] = {column: directions.get(column) or 0 for column in grid.columns}
        self.constraint: GridConstraint = GridConstraint(grid=grid, monotonicities=self.monotonicities(grid, grid))
        if classification:
            self.metrics: List[Metric] = [
                CrossEntropy(name='loss'),
                Accuracy(name='metric')
            ]
        else:
            self.metrics: List[Metric] = [
                MSE(name='loss'),
                R2(name='metric')
            ]

    def _plot(self, model):
        """Implements the plotting routine."""
        raise NotImplementedError(not_implemented_message(name='_summary_plot'))

    def scalers(self) -> Tuple[Optional[Scaler], Optional[Scaler]]:
        """Returns the dataset scalers."""
        return Scaler(default_method='std'), None if self.classification else Scaler(default_method='norm')

    def folds(self, num_folds: Optional[int] = None, **kwargs) -> Union[Fold, List[Fold]]:
        """Gets the data split in folds.

        With num_folds = None directly returns a tuple with train/test splits and scalers.
        With num_folds = 1 returns a list with a single tuple with train/val/test splits and scalers.
        With num_folds > 1 returns a list of tuples with train/val/test splits and their respective scalers.
        """
        stratify = self.train[self.label] if self.classification else None
        if num_folds is None:
            validation = dict(train=self.train, test=self.test)
            return Fold(data=self.train, label=self.label, validation=validation)
        elif num_folds == 1:
            fold = split_dataset(self.train, test_size=0.2, val_size=0.0, stratify=stratify, **kwargs)
            fold['validation'] = fold.pop('test')
            fold['test'] = self.test
            return [Fold(data=fold['train'], label=self.label, validation=fold)]
        else:
            folds = cross_validate(self.train, num_folds=num_folds, stratify=stratify, **kwargs)
            return [Fold(data=f['train'], validation={**f, 'test': self.test}, label=self.label) for f in folds]

    def monotonicities(self,
                       samples,
                       references,
                       return_matrix: bool = False) -> Union[np.ndarray, List[Tuple[int, int]]]:
        """Implements the strategy to compute expected monotonicities by returning either a list of tuples (hi, li),
        with <hi> being the higher index and <li> being the lower index, or a NxM matrix, with N the number of samples
        and M the number of references, where each cell is filled with -1, 0, or 1 depending on the kind of expected
        monotonicity between samples[i] and references[j].

        - 'samples' is the array of data points.
        - 'references' is the array of reference data point(s).
        """
        assert samples.ndim <= 2, f"'samples' should have 2 dimensions at most, but it has {samples.ndim}"
        assert references.ndim <= 2, f"'references' should have 2 dimensions at most, but it has {references.ndim}"
        # convert vectors into a matrices
        samples, references = np.atleast_2d(samples), np.atleast_2d(references)
        # increase samples dimension to match references
        samples = np.hstack([samples] * len(references)).reshape((len(samples), len(references), -1))
        # compute differences between samples to get the number of different attributes
        differences = samples - references
        num_differences = np.sign(np.abs(differences)).sum(axis=-1)
        # get whole monotonicity (sum of monotonicity signs) and mask for pairs with just one different attribute
        # (directions is converted to an array that says whether the monotonicity is increasing, decreasing, or null)
        directions = np.array([d for d in self.directions.values()])
        monotonicities = np.sign(directions * differences).sum(axis=-1).astype('int')
        monotonicities = monotonicities * (num_differences == 1)
        if return_matrix:
            # if the matrix is needed, and handle the case in which there is a single sample and a single reference,
            # since numpy.sum(axis=-1) will return a zero-dimensional array instead of a scalar
            monotonicities = np.squeeze(monotonicities)
            return np.int32(monotonicities) if monotonicities.ndim == 0 else monotonicities
        else:
            # otherwise we compute the list of indices
            return [(hi, li) for hi, row in enumerate(monotonicities) for li, mono in enumerate(row) if mono == 1]

    def summary(self, model, plot: bool = True, **data: Tuple[Any, np.ndarray]):
        """Executes the an evaluation summary on the dataset.

        - 'model' is the machine learning model used.
        - 'plot' is True if the evaluation plot is needed, False otherwise.
        - 'data' is a dictionary of data splits, if not empty, is used to print final metrics evaluations.
        """
        if plot:
            self._plot(model=model)
            plt.suptitle(model.__name__)
            plt.show()
        if len(data) > 0:
            evaluation = pd.DataFrame(index=[m.__name__ for m in self.metrics])
            for split, (x, y) in data.items():
                evaluation[split] = [metric(x, y, model.predict(x)) for metric in self.metrics]
            print('METRICS:')
            print(evaluation)
            print()
        print('CONSTRAINTS:')
        print(pd.Series(self.constraint(model=model), name='monotonicity'))


class AnalysisCallback(Callback):
    """Template callback for results analysis and plotting.

    - 'sorting_attribute' is string representing the (optional) attribute on which to sort the output data.
    - 'file_signature' is the (optional) file signature without extension where to store the output data.
    - 'num_columns' is the (optional) number of columns in the final plot, or 'auto'. If None, does not plot results.
    """

    def __init__(self,
                 sorting_attribute: Optional[str] = None,
                 file_signature: Optional[str] = None,
                 num_columns: Union[int, str] = 'auto'):
        super(AnalysisCallback, self).__init__()
        self.sorting_attribute: Optional[str] = sorting_attribute
        self.file_signature: Optional[str] = file_signature
        self.num_columns: Optional[int] = num_columns
        self.data: Optional[pd.DataFrame] = None
        self.iterations: List[Any] = []

    def on_pretraining_start(self, macs, x, y, val_data):
        self.on_iteration_start(macs, x, y, val_data)
        self.on_training_start(macs, x, y, val_data)

    def on_pretraining_end(self, macs, x, y, p, val_data):
        self.on_training_end(macs, x, y, p, val_data)
        self.on_iteration_end(macs, x, y, val_data)

    def on_iteration_end(self, macs, x, y, val_data):
        self.iterations.append(macs.iteration)

    def on_process_end(self, macs, val_data):
        # sort values
        if self.sorting_attribute is not None:
            self.data = self.data.sort_values(self.sorting_attribute)
        # write on files
        if self.file_signature is not None:
            set_pandas_options()
            self.data.to_csv(f'{self.file_signature}.csv', index_label='index')
            with open(f'{self.file_signature}.txt', 'w') as f:
                f.write(str(self.data))
        # plots results
        if self.num_columns is not None:
            plt.figure(figsize=(16, 9), tight_layout=True)
            if self.num_columns == 'auto':
                self.num_columns = self._auto_columns(ratio=16 / 9)
            num_rows = np.ceil(len(self.iterations) / self.num_columns).astype(int)
            ax = None
            for idx, it in enumerate(self.iterations):
                ax = plt.subplot(num_rows, self.num_columns, idx + 1, sharex=ax, sharey=ax)
                title = self._plot_function(it)
                ax.set(xlabel='', ylabel='')
                ax.set_title(f'{it})' if title is None else title)
            plt.show()

    def _auto_columns(self, ratio: float) -> int:
        """Implements the strategy to compute the optimal number of columns."""
        return max(np.sqrt(ratio * len(self.iterations)).round().astype(int), 1)

    def _plot_function(self, iteration: int) -> Optional[str]:
        """Implements the plotting strategy for each iteration."""
        raise NotImplementedError(not_implemented_message(name='_plot_function', static=True))


class DistanceAnalysis(AnalysisCallback):
    """Investigates the distance between ground truths, predictions, and the adjusted targets during iterations."""

    def on_process_start(self, macs, x, y, val_data):
        self.data = x.reset_index(drop=True)
        self.data['y'] = pd.Series(y, name='y')

    def on_training_end(self, macs, x, y, p, val_data):
        self.data[f'pred {macs.iteration}'] = p

    def on_adjustment_end(self, macs, x, y, z, val_data):
        self.data[f'adj {macs.iteration}'] = z

    def _auto_columns(self, ratio: float) -> int:
        """Implements the strategy to compute the optimal number of columns."""
        return super(DistanceAnalysis, self)._auto_columns(ratio=ratio / 3)

    def _plot_function(self, iteration: int) -> Optional[str]:
        x = np.arange(len(self.data))
        y, p = self.data['y'].values, self.data[f'pred {iteration}'].values
        plt.scatter(x=x, y=y, color='black', marker='_')
        plt.scatter(x=x, y=p, color='red', marker='_')
        if iteration == 0:
            plt.legend(['labels', 'predictions'])
            for i in x:
                plt.plot([i, i], [p[i], y[i]], c='black', alpha=0.4)
            avg_pred_distance, avg_label_distance = np.nan, np.nan
        else:
            j = self.data[f'adj {iteration}'].values
            plt.scatter(x=x, y=j, color='blue', alpha=0.4)
            plt.legend(['labels', 'predictions', 'adjusted'])
            for i in x:
                plt.plot([i, i], [p[i], j[i]], c='red', alpha=0.6)
                plt.plot([i, i], [y[i], j[i]], c='black', alpha=0.6)
            avg_pred_distance, avg_label_distance = np.abs(p - j).mean(), np.abs(y - j).mean()
        plt.xticks([])
        return f'{iteration}) pred. distance = {avg_pred_distance:.4f}, label distance = {avg_label_distance:.4f}'
