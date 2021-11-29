"""Preprocessing Tests."""

import unittest
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, MaxAbsScaler

from src.datasets import AbstractManager, IrisManager, WineManager, ShuttleManager, DotaManager
from src.util.preprocessing import Scaler

DATA_MANAGERS: Dict[str, AbstractManager] = {
    'iris': IrisManager(filepath='../../res/iris.csv'),
    'redwine': WineManager(filepath='../../res/redwine.csv'),
    'whitewine': WineManager(filepath='../../res/whitewine.csv'),
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

            # build and fit custom scaler for the x values
            custom_scaler = Scaler(methods=method)
            x_custom = custom_scaler.fit_transform(x)
            # build appropriate scikit scalers depending on the method
            if method == 'std':
                scikit_scaler = StandardScaler()
            elif method == 'norm':
                scikit_scaler = MinMaxScaler()
            elif method == 'zeromax':
                scikit_scaler = MaxAbsScaler()
            else:
                raise ValueError(f"'{method}' is not a valid method")
            # fit the scikit scaler and check equality between the two transformed dataframes
            x_scikit = scikit_scaler.fit_transform(x)
            msg = str(pd.concat((x_custom.reset_index(drop=True), pd.DataFrame(x_scikit)), axis=1))
            self.assertTrue(np.allclose(x_scikit, x_custom, atol=0.01), msg=msg)
            # check inverse scaling
            x_reversed = custom_scaler.inverse_transform(x_custom)
            msg = str(pd.concat((x.reset_index(drop=True), pd.DataFrame(x_reversed.reset_index(drop=True))), axis=1))
            self.assertTrue(np.allclose(x, x_reversed, atol=0.01), msg=msg)

            # build and fit custom scaler for the y values
            custom_scaler = Scaler(methods='onehot')
            y_custom = custom_scaler.fit_transform(y)
            # check name and number of classes
            expected, actual = set(y.unique()), set(y_custom.columns)
            self.assertSetEqual(expected, actual, msg=f'Expected: {expected}, actual: {actual}')
            y_scikit = OneHotEncoder(sparse=False).fit_transform(y.astype(str).values.reshape((-1, 1)))
            msg = str(pd.concat((y_custom.reset_index(drop=True), pd.DataFrame(y_scikit)), axis=1))
            self.assertTrue(np.allclose(y_scikit, y_custom), msg=msg)
            # check inverse scaling
            y_reversed = custom_scaler.inverse_transform(y_custom)
            msg = str(pd.concat((y.reset_index(drop=True), y_reversed.reset_index(drop=True)), axis=1))
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
