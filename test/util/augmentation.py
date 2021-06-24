"""Augmentation Tests."""
import unittest
from typing import Optional, Callable

import numpy as np
import pandas as pd

from src.util.augmentation import augment_data, compute_numeric_monotonicities

SEED = 0
X = pd.DataFrame.from_dict(dict(a=[1.0, 2.0, 3.0, 4.0], b=[-1.0, -2.0, -3.0, -4.0]))
Y = pd.Series(X['a'] - X['b'], name='label')
SAMPLING_FUNCTIONS = dict(
    a=(5, lambda n: np.random.randint(0, 10, size=n)),
    b=(5, lambda n: np.random.randint(-10, 0, size=n)),
)
GROUND_INDICES = [0, 1, 2, 3] + [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10


class TestAugmentation(unittest.TestCase):
    def _test(self, mono_fn: Optional[Callable]):
        np.random.seed(SEED)
        x, y = augment_data(X, Y, SAMPLING_FUNCTIONS, mono_fn)
        self.assertDictEqual(dict(x.dtypes.to_dict()), dict(X.dtypes.to_dict()))
        self.assertDictEqual(dict(y.dtypes.to_dict()), {
            Y.name: Y.dtype,
            'ground_index': int,
            'monotonicity': float
        })
        self.assertListEqual(list(y['ground_index']), GROUND_INDICES)
        for index in range(len(x)):
            xs, ys = x.iloc[index], y.iloc[index]
            ground = int(ys['ground_index'])
            # noinspection PyPep8Naming
            Xs, Ys = X.iloc[ground], Y.iloc[ground]
            # check values
            if index < len(X):
                self.assertListEqual(list(xs), list(Xs))
                self.assertEqual(ys[Y.name], Ys)
            else:
                self.assertTrue(np.any([xs[c] == Xs[c] for c in X.columns]))
                self.assertTrue(np.isnan(ys[Y.name]))
            # check monotonicities
            if mono_fn is None:
                self.assertTrue(np.isnan(ys['monotonicity']))
            else:
                real_monotonicity = mono_fn(xs, Xs)
                self.assertEqual(ys['monotonicity'], real_monotonicity)

    def test_without_monotonicities(self):
        self._test(None)

    def test_with_monotonicities(self):
        def _mono_fn(samples: np.ndarray, references: np.ndarray) -> np.ndarray:
            directions = np.array([1, -1])
            return compute_numeric_monotonicities(samples=samples, references=references, directions=directions)

        self._test(_mono_fn)
