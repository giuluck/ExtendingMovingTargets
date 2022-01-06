from typing import Optional, List

import numpy as np
import pandas as pd
from moving_targets.metrics import DIDI, CrossEntropy, Accuracy, MSE, R2
from moving_targets.util.typing import Number
from sklearn.preprocessing import OneHotEncoder

from src.datasets import AbstractManager
from src.util.cleaning import get_top_features
from src.util.preprocessing import split_dataset, Scaler, Scalers


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
        df = pd.concat((numerical_features, categorical_features, protected), axis=1).astype(float)
        df[AdultManager.TARGET] = target.astype('category').cat.codes
        return split_dataset(df, stratify=df[AdultManager.TARGET], test_size=test_size, val_size=0.0)

    def __init__(self, filepath: str, test_size: Number = 0.2):
        """
        :param filepath:
            The dataset filepath.

        :param test_size:
            The amount of total samples to be considered for training (either relative or absolute).
        """
        didi = DIDI(classification=True, protected=AdultManager.PROTECTED, percentage=True, name='constraint')
        super().__init__(filepath=filepath,
                         test_size=test_size,
                         label=AdultManager.TARGET,
                         stratify=AdultManager.TARGET,
                         x_scaling='std',
                         y_scaling=None,
                         metrics=[CrossEntropy(name='loss'), Accuracy(name='metric'), didi])

    def get_scalers(self) -> Scalers:
        # the x scaler uses the given default method for numeric features while the categorical ones (which have an
        # underscore in the name) are not scaled
        custom_methods = {c: None for c in self.train_data[0].columns if '_' in c}
        x_scaler = None if self.x_scaling is None else Scaler(default_method=self.x_scaling, **custom_methods)
        y_scaler = None if self.y_scaling is None else Scaler(default_method=self.y_scaling)
        return x_scaler, y_scaler


class CommunitiesManager(AbstractManager):
    """Data Manager for the communities dataset."""

    TARGET: str = 'violentPerPop'
    """Name of the target feature."""

    PROTECTED: str = 'race'
    """Name of the protected feature."""

    # noinspection SpellCheckingInspection
    IGNORED: List[str] = ['fold', 'state', 'murders', 'murdPerPop', 'rapes', 'rapesPerPop', 'robberies', 'robbbPerPop',
                          'communityname', 'assaults', 'assaultPerPop', 'burglaries', 'burglPerPop', 'larcenies',
                          'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop', 'nonViolPerPop']
    """List of ignored features."""

    @staticmethod
    def load_data(filepath: str, test_size: Number, max_nan: Number, n_features: Optional[int]) -> AbstractManager.Data:
        assert max_nan >= 0, "'max_nan' should be a positive value, either a float or an integer"
        df = pd.read_csv(filepath, header=0, sep=';', na_values='?', skipinitialspace=True)
        max_nan = max_nan if isinstance(max_nan, int) else int(max_nan * len(df))
        # drop rows with no associated attribute to be predicted, then build a set of ignored features which are either
        # in the default set or have too many missing values, and finally remove rows with missing values and cast
        df = df.dropna(subset=[CommunitiesManager.TARGET], axis=0).reset_index(drop=True)
        ignored = set(CommunitiesManager.IGNORED)
        for c in df.columns:
            num_nan = np.sum(df[c].isna())
            if num_nan > max_nan and c not in [CommunitiesManager.TARGET, CommunitiesManager.PROTECTED]:
                ignored.add(c)
        df = df.drop(list(ignored), axis=1).dropna(axis=0).reset_index(drop=True).astype(float)
        # perform feature selection on dataframe without protected feature (which is added at then end)
        x, y = df.drop(columns=[CommunitiesManager.PROTECTED, CommunitiesManager.TARGET]), df[
            [CommunitiesManager.TARGET]]
        ft = get_top_features(x=x, y=y, n=n_features) + [CommunitiesManager.PROTECTED, CommunitiesManager.TARGET]
        return split_dataset(df[ft], test_size=test_size, val_size=0.0)

    def __init__(self, filepath: str, test_size: Number = 0.2, max_nan: Number = 0.05, n_features: Optional[int] = 15):
        """
        :param filepath:
            The dataset filepath.

        :param test_size:
            The amount of total samples to be considered for training (either relative or absolute).

        :param max_nan:
            The maximal amount of missing values for a column to be maintained (either relative or absolute).

        :param n_features:
            The number of input features to be extracted from the cleaned dataset by importance.
        """
        didi = DIDI(classification=False, protected=CommunitiesManager.PROTECTED, percentage=True, name='constraint')
        super().__init__(filepath=filepath,
                         test_size=test_size,
                         max_nan=max_nan,
                         n_features=n_features,
                         label=CommunitiesManager.TARGET,
                         stratify=None,
                         x_scaling='std',
                         y_scaling='norm',
                         metrics=[MSE(name='loss'), R2(name='metric'), didi])

    def get_scalers(self) -> Scalers:
        # the x scaler uses the given default method for each feature but the protected one
        # (which is a categorical binary feature, thus it needs not scaling)
        custom_methods = {CommunitiesManager.PROTECTED: None}
        x_scaler = None if self.x_scaling is None else Scaler(default_method=self.x_scaling, **custom_methods)
        y_scaler = None if self.y_scaling is None else Scaler(default_method=self.y_scaling)
        return x_scaler, y_scaler
