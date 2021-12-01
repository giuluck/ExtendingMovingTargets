"""Data Manager."""

from typing import List, Tuple, Dict, Union, Optional

import numpy as np
import pandas as pd

from moving_targets.metrics import Metric
from moving_targets.util.typing import Vector, Matrix, Dataset
from src.util.preprocessing import Scaler, Scalers, split_dataset, cross_validate
from src.util.typing import Method


class AbstractManager:
    """Abstract dataset manager."""

    Data = Dict[str, pd.DataFrame]
    """Dictionary type that associates to each split name the respective dataframe."""

    DataInfo = Union[Tuple[Dataset, Scalers], List[Tuple[Dataset, Scalers]]]
    """Either a tuple of `Dataset` and `Scalers` or a list of them."""

    @staticmethod
    def load_data(**data_kwargs) -> Data:
        """Loads the dataset.

        :param data_kwargs:
            Any dataset-dependent argument that may be necessary in the implementation of this method.

        :return:
            A dictionary of dataframes representing the train and test sets, respectively.
        """
        raise NotImplementedError("Please implement abstract static method 'load_data'")

    def __init__(self, label: str, stratify: Union[None, str, List[str]], metrics: List[Metric], x_scaling: Method,
                 y_scaling: Method, **load_data_kwargs):
        """
        :param label:
            Name of the y feature.

        :param stratify:
            The feature(s) to use for stratification the dataset when splitting. If None, no stratification.

        :param metrics:
            List of `Metric` objects containing the evaluation metrics.

        :param x_scaling:
            X scaling methods.

        :param y_scaling:
            Y scaling method.

        :param load_data_kwargs:
            Any dataset-dependent argument to be passed to the static `load_data()` function.
        """
        train, test = self.load_data(**load_data_kwargs).values()
        self.train_data: Tuple[pd.DataFrame, pd.Series] = (train.drop(columns=label), train[label])
        """The training data in the form of a tuple (xtr, ytr)."""

        self.test_data: Tuple[pd.DataFrame, pd.Series] = (test.drop(columns=label), test[label])
        """The test data in the form of a tuple (xts, yts)."""

        self.stratify: Union[None, Vector, Matrix] = None if stratify is None else train[stratify]
        """The train data stratification data, if present."""

        self.metrics: List[Metric] = metrics
        """Dictionary containing the evaluation metrics indexed by name."""

        self.x_scaling: Method = x_scaling
        """X scaling methods."""

        self.y_scaling: Method = y_scaling
        """Y scaling method."""

    def get_scalers(self, x: Matrix, y: Vector) -> Tuple[Optional[Scaler], Optional[Scaler]]:
        """Returns the dataset scalers.

        :param x:
            The input data.

        :param y:
            The output data.

        :return:
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
                fold = split_dataset(*self.train_data, test_size=0.2, val_size=0.0, stratify=self.stratify,
                                     **crossval_kwargs)
                fold['validation'] = fold.pop('test')
                folds = [fold]
            else:
                folds = cross_validate(*self.train_data, num_folds=num_folds, stratify=self.stratify, **crossval_kwargs)
            return [({**fold, 'test': self.test_data}, self.get_scalers(*fold['train'])) for fold in folds]
        else:
            raise ValueError(f"{num_folds} is not an accepted value for 'num_folds'")

    def evaluation_summary(self, model, do_print: bool = False, **data_splits: Data) -> pd.DataFrame:
        """Computes the metrics over a custom set of validation data, then builds a summary.

        :param model:
            A model object having the 'predict(x)' method.

        :param do_print:
            Whether or not to print the output summary.

        :param data_splits:
            A dictionary of named `Data` arguments.

        :return:
            Either a dictionary for the metric values or a string representing the evaluation summary.
        """
        summary = {}
        for split_name, (x, y) in data_splits.items():
            p = model.predict(x).astype(np.float64)
            summary[split_name] = {}
            for metric in self.metrics:
                summary[split_name][metric.__name__] = metric(x, y, p)
        summary = pd.DataFrame.from_dict(summary)
        if do_print:
            print(summary)
        return summary
