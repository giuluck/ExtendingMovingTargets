"""Redwine and Whitewine Data Manager."""
from typing import Dict

import pandas as pd

from moving_targets.metrics import Accuracy, CrossEntropy, ClassFrequenciesStd
from src.datasets.abstract_manager import AbstractManager
from src.util.cleaning import FeatureInfo, clean_dataframe
from src.util.preprocessing import split_dataset


class WineManager(AbstractManager):
    """Data Manager for the Iris Dataset."""

    # noinspection DuplicatedCode
    FEATURES: Dict[str, FeatureInfo] = {
        'fixed acidity': FeatureInfo(dtype='float', alias='fixed_acidity'),
        'volatile acidity': FeatureInfo(dtype='float', alias='volatile_acidity'),
        'citric acid': FeatureInfo(dtype='float', alias='citric_acid'),
        'residual sugar': FeatureInfo(dtype='float', alias='residual_sugar'),
        'chlorides': FeatureInfo(dtype='float', alias='chlorides'),
        'free sulfur dioxide': FeatureInfo(dtype='float', alias='free_sulfur_dioxide'),
        'total sulfur dioxide': FeatureInfo(dtype='float', alias='total_sulfur_dioxide'),
        'density': FeatureInfo(dtype='float', alias='density'),
        'pH': FeatureInfo(dtype='float', alias='pH'),
        'sulphates': FeatureInfo(dtype='float', alias='sulphates'),
        'alcohol': FeatureInfo(dtype='float', alias='alcohol'),
        'quality': FeatureInfo(dtype='category', alias='quality')
    }
    """The redwine dataset features."""

    @staticmethod
    def load_data(filepath: str) -> AbstractManager.Data:
        df = pd.read_csv(filepath, sep=';')
        df = clean_dataframe(df, WineManager.FEATURES)
        df['quality'] = df['quality'].cat.codes
        return split_dataset(df, stratify=df['quality'], test_size=0.2, val_size=0.0)

    def __init__(self, filepath: str):
        """
        :param filepath:
            The iris dataset filepath.
        """
        super().__init__(filepath=filepath, label='quality', stratify=True, x_scaling='std', y_scaling=None,
                         metrics=[CrossEntropy(name='cce'), Accuracy(name='acc'), ClassFrequenciesStd(name='std')])
