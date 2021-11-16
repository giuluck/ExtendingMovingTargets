"""Preprocessing Tests."""

import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.util.preprocessing import Scaler

RES_FOLDER: str = '../../res/'
"""The resource folder where the datasets are placed."""


class TestCustomScalers(unittest.TestCase):
    """Tests the correctness of the `Scaler` class."""

    def _test(self, filename: str, method: str, **kwargs):
        """Performs the tests.

        :param filename:
            The name of the dataset `.csv` file.

        :param method:
            The scaling method, either 'norm' or 'std'.

        :param kwargs:
            Additional parameters to be passed to `pandas.read_csv()` method.
        """
        try:
            # load dataframe
            df = pd.read_csv(f'{RES_FOLDER}{filename}.csv', **kwargs)
            df = df[[c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]]
            # build and fit custom scaler
            custom_scaler = Scaler(methods=method)
            df_custom = custom_scaler.fit_transform(df)
            if method == 'std':
                # if std scaling, check mean and standard deviation, then build an appropriate scikit standard
                self.assertAlmostEqual(df_custom.values.mean(), 0.0, msg=f'std mean is {df_custom.values.mean()}')
                self.assertAlmostEqual(df_custom.values.std(), 1.0, msg=f'std deviation is {df_custom.values.std()}')
                scikit_scaler = StandardScaler()
            elif method == 'norm':
                # if norm scaling, check min and max, then build an appropriate scikit standard
                self.assertAlmostEqual(np.min(df_custom.values), 0.0, msg=f'norm min is {np.min(df_custom.values)}')
                self.assertAlmostEqual(np.max(df_custom.values), 1.0, msg=f'norm max is {np.max(df_custom.values)}')
                scikit_scaler = MinMaxScaler()
            else:
                raise ValueError(f"'{method}' is not a valid method")
            # fit the scikit scaler and check equality between the two transformed dataframes
            df_scikit = scikit_scaler.fit_transform(df)
            msg = f'\n> AVG DIFF:\n{abs(df_custom - df_scikit).mean()}'
            msg += f'\n> MAX DIFF:\n{abs(df_custom - df_scikit).max()}'
            msg += f'\n> JOINED:\n{pd.concat((df_custom, pd.DataFrame(df_scikit)), axis=1)}'
            self.assertTrue(np.allclose(df_scikit, df_custom, atol=0.01), msg=msg)
        except Exception as exception:
            self.fail(exception)

    def test_cars_norm(self):
        """Tests normalization scaling on cars dataset."""
        self._test('cars', 'norm')

    def test_cars_std(self):
        """Tests standardization scaling on cars dataset."""
        self._test('cars', 'std')

    def test_iris_norm(self):
        """Tests normalization scaling on iris dataset."""
        self._test('iris', 'norm')

    def test_iris_std(self):
        """Tests standardization scaling on iris dataset."""
        self._test('iris', 'std')

    def test_puzzles_norm(self):
        """Tests normalization scaling on puzzles dataset."""
        self._test('puzzles', 'norm')

    def test_puzzles_std(self):
        """Tests standardization scaling on puzzles dataset."""
        self._test('puzzles', 'std')

    def test_redwine_norm(self):
        """Tests normalization scaling on redwine dataset."""
        self._test('redwine', 'norm', sep=';')

    def test_redwine_std(self):
        """Tests standardization scaling on redwine dataset."""
        self._test('redwine', 'std', sep=';')

    def test_whitewine_norm(self):
        """Tests normalization scaling on whitewine dataset."""
        self._test('whitewine', 'norm', sep=';')

    def test_whitewine_std(self):
        """Tests standardization scaling on whitewine dataset."""
        self._test('whitewine', 'std', sep=';')
