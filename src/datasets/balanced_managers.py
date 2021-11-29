"""Data Managers for datasets related to the class balancing task."""

from abc import ABC

import pandas as pd

from moving_targets.metrics import CrossEntropy, Accuracy, ClassFrequenciesStd
from src.datasets import AbstractManager
from src.util.preprocessing import split_dataset


class BalancedCountsManager(AbstractManager, ABC):
    """Data Manager for the balanced counts dataset."""

    @staticmethod
    def _load_data(df: pd.DataFrame) -> AbstractManager.Data:
        """Processes the dataset in order to assign categorical values to che labels.

        :param df:
            The previously loaded dataframe.

        :return:
            A dictionary of dataframes representing the train and test sets, respectively.
        """
        df['class'] = df['class'].astype('category').cat.codes
        return split_dataset(df, stratify=df['class'], test_size=0.2, val_size=0.0)

    def __init__(self, filepath: str):
        """
        :param filepath:
            The dataset filepath.
        """
        super().__init__(filepath=filepath, label='class', stratify=True, x_scaling='std', y_scaling=None,
                         metrics=[CrossEntropy(name='cce'), Accuracy(name='acc'), ClassFrequenciesStd(name='std')])


class IrisManager(BalancedCountsManager):
    """Data Manager for the Iris Dataset."""

    @staticmethod
    def load_data(filepath: str) -> AbstractManager.Data:
        df = pd.read_csv(filepath)
        return BalancedCountsManager._load_data(df=df)


class WineManager(BalancedCountsManager):
    """Data Manager for the Redwine and Whitewine Datasets."""

    @staticmethod
    def load_data(filepath: str) -> AbstractManager.Data:
        df = pd.read_csv(filepath, sep=';')
        df = df.rename(columns={'quality': 'class'})
        return BalancedCountsManager._load_data(df=df)


class ShuttleManager(BalancedCountsManager):
    """Data Manager for the Shuttle Dataset."""

    @staticmethod
    def load_data(filepath: str) -> AbstractManager.Data:
        df = pd.read_csv(filepath, sep=' ', header=None, names=['time'] + [str(i) for i in range(1, 9)] + ['class'])
        return BalancedCountsManager._load_data(df=df)


class DotaManager(BalancedCountsManager):
    """Data Manager for the Dota2 Dataset."""

    @staticmethod
    def load_data(filepath: str) -> AbstractManager.Data:
        # The first column (0) represents the category, the columns 1, 2, and 3 represent the cluster id (they are
        # discarded for simplicity), and the next columns (from 4 to 116) represent the integer data features.
        df = pd.read_csv(filepath, header=None, names=[str(i) for i in range(117)])
        df = df.drop(columns=['1', '2', '3'])
        df = df.rename(columns={'0': 'class'})
        return BalancedCountsManager._load_data(df=df)
