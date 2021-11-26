"""Preprocessing Tests."""

import unittest
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, MaxAbsScaler

from src.datasets import AbstractManager, IrisManager, WineManager
from src.util.preprocessing import Scaler

DATA_MANAGERS: Dict[str, AbstractManager] = {
    'iris': IrisManager(filepath='../../res/iris.csv'),
    'redwine': WineManager(filepath='../../res/redwine.csv'),
    'whitewine': WineManager(filepath='../../res/whitewine.csv')
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
            if method == 'std':
                # if std scaling, check mean and standard deviation, then build an appropriate scikit standard
                self.assertAlmostEqual(x_custom.values.mean(), 0.0, msg=f'std mean is {x_custom.values.mean()}')
                self.assertAlmostEqual(x_custom.values.std(), 1.0, msg=f'std deviation is {x_custom.values.std()}')
                scikit_scaler = StandardScaler()
            elif method == 'norm':
                # if norm scaling, check min and max, then build an appropriate scikit standard
                self.assertAlmostEqual(np.min(x_custom.values), 0.0, msg=f'norm min is {np.min(x_custom.values)}')
                self.assertAlmostEqual(np.max(x_custom.values), 1.0, msg=f'norm max is {np.max(x_custom.values)}')
                scikit_scaler = MinMaxScaler()
            elif method == 'zeromax':
                # if zeromax scaling, check max, then build an appropriate scikit standard
                self.assertAlmostEqual(np.max(x_custom.values), 1.0, msg=f'norm max is {np.max(x_custom.values)}')
                scikit_scaler = MaxAbsScaler()
            else:
                raise ValueError(f"'{method}' is not a valid method")
            # fit the scikit scaler and check equality between the two transformed dataframes
            x_scikit = scikit_scaler.fit_transform(x)
            msg = str(pd.concat((x_custom, pd.DataFrame(x_scikit)), axis=1))
            self.assertTrue(np.allclose(x_scikit, x_custom, atol=0.01), msg=msg)
            # check inverse scaling
            x_reversed = custom_scaler.inverse_transform(x_custom)
            msg = str(pd.concat((x, pd.DataFrame(x_reversed)), axis=1))
            self.assertTrue(np.allclose(x, x_reversed, atol=0.01), msg=msg)

            # build and fit custom scaler for the y values
            custom_scaler = Scaler(methods='onehot')
            y_custom = custom_scaler.fit_transform(y)
            # check name and number of classes
            expected, actual = set(y.unique()), set(y_custom.columns)
            self.assertSetEqual(expected, actual, msg=f'Expected: {expected}, actual: {actual}')
            y_scikit = OneHotEncoder().fit_transform(y.astype(str).values.reshape((-1, 1))).todense()
            msg = str(pd.concat((y_custom, pd.DataFrame(y_scikit)), axis=1))
            self.assertTrue(np.allclose(y_scikit, y_custom), msg=msg)
            # check inverse scaling
            y_reversed = custom_scaler.inverse_transform(y_custom)
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
