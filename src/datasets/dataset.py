"""Data Manager."""
from typing import Dict, Optional, Tuple, List, Union

import numpy as np
import pandas as pd
from moving_targets.metrics import MSE, R2, CrossEntropy, Accuracy, Metric
from moving_targets.util.errors import not_implemented_message

from src.util.preprocessing import Scaler, split_dataset, cross_validate


class Fold:
    """Data class containing the information of a fold for k-fold cross-validation."""

    def __init__(self,
                 data: pd.DataFrame,
                 label: str,
                 x_scaler: Scaler,
                 y_scaler: Scaler,
                 validation: Dict[str, pd.DataFrame]):
        """
        :param data:
            The training data.

        :param label:
            The target label.

        :param x_scaler:
            The input data scaler.

        :param y_scaler:
            The output data scaler.

        :param validation:
            A shared validation dataset which is common among all the k folds.
        """

        def split_df(df: pd.DataFrame, fit: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
            x_df, y_df = df.drop(columns=label), df[label].values
            if fit:
                x_scaler.fit(x_df)
                y_scaler.fit(y_df)
            return x_scaler.transform(x_df), y_scaler.transform(y_df)

        self.x, self.y = split_df(df=data, fit=True)
        self.x_scaler, self.y_scaler = x_scaler, y_scaler
        self.validation: Dict[str, Tuple[pd.DataFrame, np.ndarray]] = {k: split_df(df=v) for k, v in validation.items()}


class Dataset:
    """Abstract Dataset Manager."""

    @staticmethod
    def load() -> Dict[str, pd.DataFrame]:
        """Loads the dataset.

        :return:
            A dictionary of dataframes representing the train and test sets, respectively.
        """
        raise NotImplementedError(not_implemented_message(name='load_data', static=True))

    def __init__(self,
                 label: str,
                 directions: Dict[str, int],
                 grid: pd.DataFrame,
                 classification: bool):
        train, test = self.load().values()

        self.train: pd.DataFrame = train
        """The training data."""

        self.test: pd.DataFrame = test
        """The test data."""

        self.label: str = label
        """The name of the target feature."""

        self.directions: Dict[str, int] = directions
        """A dictionary pairing each monotonic attribute to its direction (-1, 0, or 1)."""

        self.classification: bool = classification
        """Whether the model should solve a (binary) classification or a regression task."""

        # TODO: add violation metric using grid (NB: need to scale the grid... add scalers in Moving Targets?)
        self.metrics: List[Metric] = []  # ConstraintMetric(directions=directions, grid=grid)
        """The list of metrics to evaluate models learning on this dataset"""

        if classification:
            self.metrics.insert(0, Accuracy(name='metric'))
            self.metrics.insert(0, CrossEntropy(name='loss'))
        else:
            self.metrics.insert(0, R2(name='metric'))
            self.metrics.insert(0, MSE(name='loss'))

    def get_scalers(self) -> Tuple[Scaler, Scaler]:
        """Returns the dataset scalers.

        :return:
            A pair of scalers, one for the input and one for the output data, respectively.
        """
        return Scaler(default_method='std'), Scaler(default_method=None if self.classification else 'norm')

    def get_folds(self, num_folds: Optional[int] = None, **kwargs) -> Union[Fold, List[Fold]]:
        """Gets the data split in folds.

        With num_folds = None directly returns a tuple with train/test splits and scalers.
        With num_folds = 1 returns a list with a single tuple with train/val/test splits and scalers.
        With num_folds > 1 returns a list of tuples with train/val/test splits and their respective scalers.

        :param num_folds:
            The number of folds for k-fold cross-validation.

        :param kwargs:
            Arguments passed either to `split_dataset()` or `cross_validate()` method, depending on the number of folds.

        :return:
            Either a tuple of `Dataset` and `Scalers` or a list of them, depending on the number of folds.
        """
        x_scaler, y_scaler = self.get_scalers()
        stratify = self.train[self.label] if self.classification else None
        fold_kwargs = dict(label=self.label, x_scaler=x_scaler, y_scaler=y_scaler)
        if num_folds is None:
            validation = dict(train=self.train, test=self.test)
            return Fold(data=self.train, validation=validation, **fold_kwargs)
        elif num_folds == 1:
            fold = split_dataset(self.train, test_size=0.2, val_size=0.0, stratify=stratify, **kwargs)
            fold['validation'] = fold.pop('test')
            fold['test'] = self.test
            return [Fold(data=fold['train'], validation=fold, **fold_kwargs)]
        else:
            folds = cross_validate(self.train, num_folds=num_folds, stratify=stratify, **kwargs)
            return [Fold(data=fold['train'], validation={**fold, 'test': self.test}, **fold_kwargs) for fold in folds]
