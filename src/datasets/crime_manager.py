from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from moving_targets.metrics import MSE, R2
from moving_targets.util.typing import Number, Matrix, Vector
from src.datasets import AbstractManager
from src.util.cleaning import get_top_features
from src.util.preprocessing import split_dataset, Scaler


class CrimeManager(AbstractManager):
    """Data Manager for the crime dataset."""

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
        df = df.dropna(subset=[CrimeManager.TARGET], axis=0).reset_index(drop=True)
        ignored = set(CrimeManager.IGNORED)
        for c in df.columns:
            num_nan = np.sum(df[c].isna())
            if num_nan > max_nan and c not in [CrimeManager.TARGET, CrimeManager.PROTECTED]:
                ignored.add(c)
        df = df.drop(list(ignored), axis=1).dropna(axis=0).reset_index(drop=True).astype(float)
        # perform feature selection on dataframe without protected feature (which is added at then end)
        x, y = df.drop(columns=[CrimeManager.PROTECTED, CrimeManager.TARGET]), df[[CrimeManager.TARGET]]
        ft = get_top_features(x=x, y=y, n=n_features) + [CrimeManager.PROTECTED, CrimeManager.TARGET]
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
        super().__init__(filepath=filepath,
                         test_size=test_size,
                         max_nan=max_nan,
                         n_features=n_features,
                         label=CrimeManager.TARGET,
                         stratify=None,
                         x_scaling='std',
                         y_scaling='norm',
                         metrics=[MSE(name='mse'), R2(name='r2')])

    def get_scalers(self, x: Matrix, y: Vector) -> Tuple[Optional[Scaler], Optional[Scaler]]:
        # the x scaler uses the given default method for each feature but the protected one
        # (which is a categorical binary feature, thus it needs not scaling)
        return Scaler(default_method=self.x_scaling, **{CrimeManager.PROTECTED: None}).fit(x), None
