"""Data Managers for datasets related to the class balancing task."""

from abc import ABC

import pandas as pd
from moving_targets.metrics import CrossEntropy, Accuracy, ClassFrequenciesStd
from moving_targets.util.typing import Number

from src.datasets import AbstractManager
from src.util.preprocessing import split_dataset


class BalancedCountsManager(AbstractManager, ABC):
    """Data Manager for the balanced counts task datasets."""

    @staticmethod
    def _load_data(df: pd.DataFrame, test_size: Number) -> AbstractManager.Data:
        """Processes the dataset in order to assign categorical values to che labels.

        :param df:
            The previously loaded dataframe.

        :param test_size:
            The test size.

        :return:
            A dictionary of dataframes representing the train and test sets, respectively.
        """
        df['class'] = df['class'].astype('category').cat.codes
        return split_dataset(df, stratify=df['class'], test_size=test_size, val_size=0.0)

    def __init__(self, filepath: str, test_size: Number = 0.2):
        """
        :param filepath:
            The dataset filepath.

        :param test_size:
            The amount of total samples to be considered for training (either relative or absolute).
        """
        super().__init__(filepath=filepath,
                         test_size=test_size,
                         label='class',
                         stratify='class',
                         x_scaling='std',
                         y_scaling=None,
                         metrics=[
                             CrossEntropy(name='loss'),
                             Accuracy(name='metric'),
                             ClassFrequenciesStd(name='constraint')
                         ])


class IrisManager(BalancedCountsManager):
    """Data Manager for the Iris Dataset."""

    @staticmethod
    def load_data(filepath: str, test_size: Number) -> AbstractManager.Data:
        df = pd.read_csv(filepath)
        return BalancedCountsManager._load_data(df=df, test_size=test_size)


class RedwineManager(BalancedCountsManager):
    """Data Manager for the Redwine Dataset."""

    @staticmethod
    def load_data(filepath: str, test_size: Number) -> AbstractManager.Data:
        df = pd.read_csv(filepath, sep=';')
        df = df.rename(columns={'quality': 'class'})
        return BalancedCountsManager._load_data(df=df, test_size=test_size)


class WhitewineManager(RedwineManager):
    """Data Manager for the Whitewine Dataset."""


class ShuttleManager(BalancedCountsManager):
    """Data Manager for the Shuttle Dataset."""

    @staticmethod
    def load_data(filepath: str, test_size: Number) -> AbstractManager.Data:
        df = pd.read_csv(filepath, sep=' ', header=None, names=['time'] + [str(i) for i in range(1, 9)] + ['class'])
        return BalancedCountsManager._load_data(df=df, test_size=test_size)


class DotaManager(BalancedCountsManager):
    """Data Manager for the Dota2 Dataset."""

    @staticmethod
    def load_data(filepath: str, test_size: Number) -> AbstractManager.Data:
        # The first column (0) represents the category, the columns 1, 2, and 3 represent the cluster id (they are
        # discarded for simplicity), and the next columns (from 4 to 116) represent the integer data features.
        df = pd.read_csv(filepath, header=None, names=[str(i) for i in range(117)])
        df = df.drop(columns=['1', '2', '3'])
        df = df.rename(columns={'0': 'class'})
        return BalancedCountsManager._load_data(df=df, test_size=test_size)
