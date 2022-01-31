"""Data Manager."""
from typing import Dict, Optional, Tuple, List, Union, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from moving_targets.metrics import MSE, R2, CrossEntropy, Accuracy, Metric
from moving_targets.util.errors import not_implemented_message
from moving_targets.util.scalers import Scaler

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
