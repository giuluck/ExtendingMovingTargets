"""Preprocessing Tests."""

import unittest
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, MaxAbsScaler

from src.datasets import AbstractManager, IrisManager, RedwineManager, WhitewineManager, ShuttleManager, DotaManager
from src.util.preprocessing import Scaler

DATA_MANAGERS: Dict[str, AbstractManager] = {
    'iris': IrisManager(filepath='../../res/iris.csv'),
    'redwine': RedwineManager(filepath='../../res/redwine.csv'),
    'whitewine': WhitewineManager(filepath='../../res/whitewine.csv'),
    'shuttle': ShuttleManager(filepath='../../res/shuttle.trn'),
    'dota': DotaManager(filepath='../../res/dota2.csv')
}
"""Dictionary which links each dataset with its dataset manager."""


class TestCustomScalers(unittest.TestCase):
    """Tests the correctness of the `Scaler` class."""

    def _test(self, filename: str, method: str):
        """Performs the tests.

        :param filename:
            The name of the dataset `.csv` file.

        :param method:
            The scaling method, one in 'norm', 'std', or 'zeromax'.
        """
        try:
            x, y = DATA_MANAGERS[filename].train_data
            x, y = x.reset_index(drop=True), y.reset_index(drop=True)
            # in order to test custom methods, it creates a scaler for the whole dataframe using the passed method as
            # custom scaler (for x values) and one-hot encoding for the output labels (named as the y series)
            custom_scaler = Scaler(default_method=method, **{y.name: 'onehot'})
            df = pd.concat((x, y), axis=1)
            df_custom = custom_scaler.fit_transform(df)
            df_reversed = custom_scaler.inverse_transform(df_custom)

            # build appropriate scikit scalers for the input data depending on the method
            if method == 'std':
                scikit_scaler = StandardScaler()
            elif method == 'norm':
                scikit_scaler = MinMaxScaler()
            elif method == 'zeromax':
                scikit_scaler = MaxAbsScaler()
            else:
                raise ValueError(f"'{method}' is not a valid method")
            # check equality between the two transformed dataframes
            x_custom = df_custom[x.columns]
            x_scikit = scikit_scaler.fit_transform(x)
            msg = str(pd.concat((x_custom, pd.DataFrame(x_scikit)), axis=1))
            self.assertTrue(np.allclose(x_scikit, x_custom, atol=0.01), msg=msg)
            # check inverse scaling
            x_reversed = df_reversed[x.columns]
            msg = str(pd.concat((x, pd.DataFrame(x_reversed)), axis=1))
            self.assertTrue(np.allclose(x, x_reversed, atol=0.01), msg=msg)

            # check name and number of classes
            y_custom = df_custom.drop(columns=x.columns)
            expected, actual = set(y.unique()), set(y_custom.columns)
            self.assertSetEqual(expected, actual, msg=f'Expected: {expected}, actual: {actual}')
            y_scikit = OneHotEncoder(sparse=False).fit_transform(y.astype(str).values.reshape((-1, 1)))
            msg = str(pd.concat((y_custom, pd.DataFrame(y_scikit)), axis=1))
            self.assertTrue(np.allclose(y_scikit, y_custom), msg=msg)
            # check inverse scaling
            y_reversed = df_reversed[y.name]
            msg = str(pd.concat((y, y_reversed), axis=1))
            self.assertTrue(np.equal(y, y_reversed).all(), msg=msg)
        except Exception as exception:
            self.fail(exception)

    def test_iris_norm(self):
        """Tests normalization scaling on iris dataset."""
        self._test('iris', 'norm')

    def test_iris_std(self):
        """Tests standardization scaling on iris dataset."""
        self._test('iris', 'std')

    def test_iris_zeromax(self):
        """Tests zeromax scaling on iris dataset."""
        self._test('iris', 'zeromax')

    def test_redwine_norm(self):
        """Tests normalization scaling on redwine dataset."""
        self._test('redwine', 'norm')

    def test_redwine_std(self):
        """Tests standardization scaling on redwine dataset."""
        self._test('redwine', 'std')

    def test_redwine_zeromax(self):
        """Tests zeromax scaling on redwine dataset."""
        self._test('redwine', 'zeromax')

    def test_whitewine_norm(self):
        """Tests normalization scaling on whitewine dataset."""
        self._test('whitewine', 'norm')

    def test_whitewine_std(self):
        """Tests standardization scaling on whitewine dataset."""
        self._test('whitewine', 'std')

    def test_whitewine_zeromax(self):
        """Tests zeromax scaling on whitewine dataset."""
        self._test('whitewine', 'zeromax')

    def test_shuttle_norm(self):
        """Tests normalization scaling on shuttle dataset."""
        self._test('shuttle', 'norm')

    def test_shuttle_std(self):
        """Tests standardization scaling on shuttle dataset."""
        self._test('shuttle', 'std')

    def test_shuttle_zeromax(self):
        """Tests zeromax scaling on shuttle dataset."""
        self._test('shuttle', 'zeromax')

    def test_dota_norm(self):
        """Tests normalization scaling on dota dataset."""
        self._test('dota', 'norm')

    def test_dota_std(self):
        """Tests standardization scaling on dota dataset."""
        self._test('dota', 'std')

    def test_dota_zeromax(self):
        """Tests zeromax scaling on dota dataset."""
        self._test('dota', 'zeromax')
