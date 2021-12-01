from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from moving_targets.metrics import MSE, R2
from moving_targets.util.typing import Number, Matrix, Vector
from src.datasets import AbstractManager
from src.util.preprocessing import split_dataset, Scaler


class AdultManager(AbstractManager):
    """Data Manager for the adult dataset."""

    TARGET: str = 'income'
    """Name of the target feature."""

    PROTECTED: str = 'race'
    """Name of the protected feature."""

    IGNORED: List[str] = ['education', 'native-country']
    """List of ignored features."""

    @staticmethod
    def load_data(filepath: str, test_size: Number) -> AbstractManager.Data:
        df = pd.read_csv(filepath, header=0, sep=';', na_values='?', skipinitialspace=True)
        df = df.dropna(axis=0).reset_index(drop=True)
        # manage target and protected features
        target = pd.Series(df[AdultManager.TARGET].astype('category').cat.codes, name=AdultManager.TARGET)
        protected = pd.get_dummies(df[AdultManager.PROTECTED], prefix=AdultManager.PROTECTED)
        # manage other input features
        features = df.drop(columns=AdultManager.IGNORED + [AdultManager.PROTECTED, AdultManager.TARGET])
        encoder = OneHotEncoder(drop=None, dtype=np.float, sparse=False)
        categorical_features = encoder.fit_transform(features.select_dtypes(include=['object']))
        categorical_features = pd.DataFrame(categorical_features, columns=encoder.get_feature_names_out())
        numerical_features = features.select_dtypes(exclude=['object'])
        # obtain output data by concatenating features
        df = pd.concat((numerical_features, categorical_features, protected, target), axis=1).astype(float)
        return split_dataset(df, stratify=df[AdultManager.TARGET], test_size=test_size, val_size=0.0)

    def __init__(self, filepath: str, test_size: Number = 0.2):
        """
        :param filepath:
            The dataset filepath.

        :param test_size:
            The amount of total samples to be considered for training (either relative or absolute).
        """
        super().__init__(filepath=filepath,
                         test_size=test_size,
                         label=AdultManager.TARGET,
                         stratify=AdultManager.TARGET,
                         x_scaling='std',
                         y_scaling=None,
                         metrics=[MSE(name='mse'), R2(name='r2')])

    def get_scalers(self, x: Matrix, y: Vector) -> Tuple[Optional[Scaler], Optional[Scaler]]:
        # the x scaler uses the given default method for numeric features while the categorical ones (which have an
        # underscore in the name) are not scaled
        custom_methods = {c: None for c in x.columns if '_' in c}
        return Scaler(default_method=self.x_scaling, **custom_methods).fit(x), None
