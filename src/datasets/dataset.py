"""Data Manager."""
from typing import List, Any, Optional, Tuple, Dict, Callable
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from moving_targets.callbacks import Callback
from moving_targets.metrics import MSE, R2, CrossEntropy, Accuracy, Metric
from moving_targets.util.errors import not_implemented_message
from moving_targets.util.scalers import Scaler

from src.util.analysis import set_pandas_options
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


class Dataset:
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
        raise NotImplementedError(not_implemented_message(name='load_data', static=True))

    def __init__(self,
                 label: str,
                 directions: Dict[str, int],
                 grid: pd.DataFrame,
                 classification: bool):
        train, test = self.load().values()

        self.train: pd.DataFrame = train
        self.test: pd.DataFrame = test
        self.label: str = label
        self.directions: Dict[str, int] = directions
        self.grid: pd.DataFrame = grid
        self.classification: bool = classification

        # TODO: add violation metric using grid
        self.metrics: List[Metric] = []  # ConstraintMetric(directions=directions, grid=grid)

        if classification:
            self.metrics.insert(0, Accuracy(name='metric'))
            self.metrics.insert(0, CrossEntropy(name='loss'))
        else:
            self.metrics.insert(0, R2(name='metric'))
            self.metrics.insert(0, MSE(name='loss'))

    def _plot(self, model):
        """Implements the plotting routine."""
        raise NotImplementedError(not_implemented_message(name='_summary_plot'))

    def get_scalers(self) -> Tuple[Scaler, Scaler]:
        """Returns the dataset scalers."""
        return Scaler(default_method='std'), Scaler(default_method=None if self.classification else 'norm')

    def get_folds(self, num_folds: Optional[int] = None, **kwargs) -> Union[Fold, List[Fold]]:
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
            print(evaluation)


class AnalysisCallback(Callback):
    """Template callback for results analysis and plotting.

    - 'sorting_attribute' is string representing the (optional) attribute on which to sort the output data.
    - 'file_signature' is the (optional) file signature without extension where to store the output data.
    - 'num_columns' is the (optional) number of columns in the final plot, or 'auto'. If None, does not plot results.
    """

    def __init__(self,
                 sorting_attribute: Optional[str] = None,
                 file_signature: Optional[str] = None,
                 num_columns: Union[int, str] = 'auto',
                 figsize: Tuple[int, int] = (16, 9),
                 tight_layout: bool = True,
                 **plt_kwargs):
        super(AnalysisCallback, self).__init__()
        self.sorting_attribute: Optional[str] = sorting_attribute
        self.file_signature: Optional[str] = file_signature
        self.num_columns: Optional[int] = num_columns
        self.plt_kwargs: Dict[str, Any] = {'figsize': figsize, 'tight_layout': tight_layout, **plt_kwargs}
        self.data: Optional[pd.DataFrame] = None
        self.iterations: List[Any] = []

    def on_pretraining_start(self, macs, x, y, val_data):
        self.on_iteration_start(macs, x, y, val_data)
        self.on_training_start(macs, x, y, val_data)

    def on_pretraining_end(self, macs, x, y, val_data):
        self.on_training_end(macs, x, y, val_data)
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
            plt.figure(**self.plt_kwargs)
            if self.num_columns == 'auto':
                self.num_columns = self._auto_columns(ratio=np.divide(*self.plt_kwargs['figsize']))
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

    def _plot_function(self, iteration: Any) -> Optional[str]:
        """Implements the plotting strategy for each iteration."""
        raise NotImplementedError(not_implemented_message(name='_plot_function', static=True))


class DistanceAnalysis(AnalysisCallback):
    """Investigates the distance between ground truths, predictions, and the adjusted targets during iterations."""

    def on_process_start(self, macs, x, y, val_data):
        self.data = x.reset_index(drop=True)
        self.data['y'] = pd.Series(y, name='y')

    def on_training_end(self, macs, x, y, val_data):
        self.data[f'pred {macs.iteration}'] = macs.predict(x)

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data):
        self.data[f'adj {macs.iteration}'] = adjusted_y

    def _auto_columns(self, ratio: float) -> int:
        """Implements the strategy to compute the optimal number of columns."""
        return super(DistanceAnalysis, self)._auto_columns(ratio=ratio / 3)

    def _plot_function(self, iteration: Any) -> Optional[str]:
        x = np.arange(len(self.data))
        y, p = self.data['y'].values, self.data[f'pred {iteration}'].values
        plt.scatter(x=x, y=p, color='red', marker='_')
        plt.scatter(x=x, y=y, color='black', marker='_')
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
