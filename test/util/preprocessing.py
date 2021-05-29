"""Preprocessing Tests."""

import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.util.preprocessing import Scaler


class TestCustomScalers(unittest.TestCase):
    def _test(self, filename, method, **kwargs):
        # check methods
        df = pd.read_csv(f'../../res/{filename}.csv', **kwargs)
        df = df[[c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]]
        custom_scaler = Scaler(methods=method)
        df_custom = custom_scaler.fit_transform(df)
        if method == 'std':
            self.assertAlmostEqual(df_custom.values.mean(), 0.0, msg=f'std mean is {df_custom.values.mean()}')
            self.assertAlmostEqual(df_custom.values.std(), 1.0, msg=f'std deviation is {df_custom.values.std()}')
            scikit_scaler = StandardScaler()
        elif method == 'norm':
            self.assertAlmostEqual(np.min(df_custom.values), 0.0, msg=f'norm min is {np.min(df_custom.values)}')
            self.assertAlmostEqual(np.max(df_custom.values), 1.0, msg=f'norm max is {np.max(df_custom.values)}')
            scikit_scaler = MinMaxScaler()
        else:
            raise ValueError(f"'{method}' is not a valid method")
        df_scikit = scikit_scaler.fit_transform(df)
        msg = f'\n> AVG DIFF:\n{abs(df_custom - df_scikit).mean()}'
        msg += f'\n> MAX DIFF:\n{abs(df_custom - df_scikit).max()}'
        msg += f'\n> JOINED:\n{pd.concat((df_custom, pd.DataFrame(df_scikit)), axis=1)}'
        self.assertTrue(np.allclose(df_scikit, df_custom, atol=0.01), msg=msg)

    def test_cars_norm(self):
        self._test('cars', 'norm')

    def test_cars_std(self):
        self._test('cars', 'std')

    def test_iris_norm(self):
        self._test('iris', 'norm')

    def test_iris_std(self):
        self._test('iris', 'std')

    def test_puzzles_norm(self):
        self._test('puzzles', 'norm')

    def test_puzzles_std(self):
        self._test('puzzles', 'std')

    def test_redwine_norm(self):
        self._test('redwine', 'norm', sep=';')

    def test_redwine_std(self):
        self._test('redwine', 'std', sep=';')

    def test_whitewine_norm(self):
        self._test('whitewine', 'norm', sep=';')

    def test_whitewine_std(self):
        self._test('whitewine', 'std', sep=';')
