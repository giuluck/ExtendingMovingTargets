"""Iris Data Manager."""
from typing import Dict

import pandas as pd

from moving_targets.metrics import Accuracy, CrossEntropy, ClassFrequenciesStd
from src.datasets.abstract_manager import AbstractManager
from src.util.cleaning import FeatureInfo, clean_dataframe
from src.util.preprocessing import split_dataset


class IrisManager(AbstractManager):
    """Data Manager for the Iris Dataset."""

    FEATURES: Dict[str, FeatureInfo] = {
        'sepal length in cm': FeatureInfo(dtype='float', alias='sepal_length'),
        'sepal width in cm': FeatureInfo(dtype='float', alias='sepal_width'),
        'petal length in cm': FeatureInfo(dtype='float', alias='petal_length'),
        'petal width in cm': FeatureInfo(dtype='float', alias='petal_width'),
        'class': FeatureInfo(dtype='category', alias='class')
    }
    """The iris dataset features."""

    @staticmethod
    def load_data(filepath: str) -> AbstractManager.Data:
        df = pd.read_csv(filepath)
        df = clean_dataframe(df, IrisManager.FEATURES)
        df['class'] = df['class'].cat.codes
        return split_dataset(df, stratify=df['class'], test_size=0.2, val_size=0.0)

    def __init__(self, filepath: str):
        """
        :param filepath:
            The iris dataset filepath.
        """
        super().__init__(filepath=filepath, label='class', stratify=True, x_scaling='std', y_scaling=None,
                         metrics=[CrossEntropy(name='cce'), Accuracy(name='acc'), ClassFrequenciesStd(name='std')])
