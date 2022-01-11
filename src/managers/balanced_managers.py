"""Data Managers for datasets related to the class balancing task."""
import importlib.resources
from typing import Dict, List

import pandas as pd
from moving_targets import MACS
from moving_targets.learners import LogisticRegression
from moving_targets.metrics import CrossEntropy, Accuracy, ClassFrequenciesStd, Metric
from moving_targets.util.errors import not_implemented_message

from src.managers.abstract_manager import AbstractManager, Config
from src.util.masters import BalancedCounts
from src.util.preprocessing import split_dataset


class BalancedCountsManager(AbstractManager):
    """Data Manager for the balanced counts task datasets."""

    @classmethod
    def _load(cls, filepath) -> pd.DataFrame:
        raise NotImplementedError(not_implemented_message(name='_load'))

    @classmethod
    def data(cls) -> Dict[str, pd.DataFrame]:
        with importlib.resources.path('res', f"{cls.name()}.csv") as filepath:
            df = cls._load(filepath)
        df['class'] = df['class'].astype('category').cat.codes
        return split_dataset(df, stratify=df['class'], test_size=0.2, val_size=0.0)

    @classmethod
    def model(cls, config: Config) -> MACS:
        return MACS(
            learner=LogisticRegression(max_iter=10000, **config.learner_kwargs),
            master=BalancedCounts(**config.master_kwargs),
            metrics=cls.metrics(),
            stats=True,
            **config.macs_kwargs
        )

    @classmethod
    def metrics(cls) -> List[Metric]:
        return [CrossEntropy(name='loss'), Accuracy(name='metric'), ClassFrequenciesStd(name='constraint')]

    def __init__(self, config: Config):
        """
        :param config:
            A configuration instance to set the experiment parameters.
        """
        super(BalancedCountsManager, self).__init__(label='class',
                                                    stratify=True,
                                                    x_scaling='std',
                                                    y_scaling=None,
                                                    config=config)


class IrisManager(BalancedCountsManager):
    """Data Manager for the Iris Dataset."""

    @classmethod
    def _load(cls, filepath) -> pd.DataFrame:
        return pd.read_csv(filepath)


class RedwineManager(BalancedCountsManager):
    """Data Manager for the Redwine Dataset."""

    @classmethod
    def _load(cls, filepath) -> pd.DataFrame:
        df = pd.read_csv(filepath, sep=';')
        return df.rename(columns={'quality': 'class'})


class WhitewineManager(BalancedCountsManager):
    """Data Manager for the Whitewine Dataset."""

    @classmethod
    def _load(cls, filepath) -> pd.DataFrame:
        df = pd.read_csv(filepath, sep=';')
        return df.rename(columns={'quality': 'class'})


class ShuttleManager(BalancedCountsManager):
    """Data Manager for the Shuttle Dataset."""

    @classmethod
    def _load(cls, filepath) -> pd.DataFrame:
        return pd.read_csv(filepath, sep=' ', header=None, names=['time'] + [str(i) for i in range(1, 9)] + ['class'])


class DotaManager(BalancedCountsManager):
    """Data Manager for the Dota2 Dataset."""

    @classmethod
    def _load(cls, filepath) -> pd.DataFrame:
        # The first column (0) represents the category, the columns 1, 2, and 3 represent the cluster id (they are
        # discarded for simplicity), and the next columns (from 4 to 116) represent the integer data features.
        df = pd.read_csv(filepath, header=None, names=[str(i) for i in range(117)])
        df = df.drop(columns=['1', '2', '3'])
        return df.rename(columns={'0': 'class'})
