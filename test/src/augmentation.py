"""Augmentation Tests."""
import unittest
from typing import Optional, Callable

import numpy as np
import pandas as pd

from src.util.augmentation import augment_data, compute_numeric_monotonicities

SEED = 0
"""The chosen random seed."""

X = pd.DataFrame.from_dict(dict(a=[1.0, 2.0, 3.0, 4.0], b=[-1.0, -2.0, -3.0, -4.0]))
"""The input variables (x1|x2).

| x1 | x2 | y |
|  1 | -1 | 2 |
|  2 | -2 | 4 |
|  3 | -3 | 6 |
|  4 | -4 | 8 |
"""

Y = pd.Series(X['a'] - X['b'], name='label')
"""The output variables (y).

| x1 | x2 | y |
|  1 | -1 | 2 |
|  2 | -2 | 4 |
|  3 | -3 | 6 |
|  4 | -4 | 8 |
"""

SAMPLING_FUNCTIONS = dict(
    a=(5, lambda n: np.random.randint(0, 10, size=n)),
    b=(5, lambda n: np.random.randint(-10, 0, size=n)),
)
"""The sampling functions of the two input variables.

- x1: int([0, 10]);
- x2: int([-10, 0]).
"""

GROUND_INDICES = [0, 1, 2, 3] + [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10
"""The ground indices for the augmentation procedure.

[0, 1, 2, 3, 0, ..., 0, 1, ..., 1, 2, ..., 2, 3, ..., 3].
"""

DIRECTIONS = np.array([1, -1])
"""The monotonicity directions.

- x1: increasing;
- x2: decreasing.
"""


class TestAugmentation(unittest.TestCase):
    """Tests the correctness of the augmentation procedures."""

    def _test(self, mono_fn: Optional[Callable]):
        """Performs the tests.

        :param mono_fn:
            Either None or a callable function to compute monotonicities.
        """
        try:
            np.random.seed(SEED)
            x, y = augment_data(X, Y, SAMPLING_FUNCTIONS, mono_fn)
            # check type equality between original and augmented data
            self.assertDictEqual(dict(x.dtypes.to_dict()), dict(X.dtypes.to_dict()))
            self.assertDictEqual(dict(y.dtypes.to_dict()), {
                Y.name: Y.dtype,
                'ground_index': int,
                'monotonicity': float
            })
            # check correctness of ground indices
            self.assertListEqual(list(y['ground_index']), GROUND_INDICES)
            for index in range(len(x)):
                xs, ys = x.iloc[index], y.iloc[index]
                ground = int(ys['ground_index'])
                # noinspection PyPep8Naming
                Xs, Ys = X.iloc[ground], Y.iloc[ground]
                # check correctness of records
                # same x/y records for non-augmented data
                if index < len(X):
                    self.assertListEqual(list(xs), list(Xs))
                    self.assertEqual(ys[Y.name], Ys)
                # at least one equal attribute in the x record and no label for augmented data
                else:
                    self.assertTrue(np.any([xs[c] == Xs[c] for c in X.columns]))
                    self.assertTrue(np.isnan(ys[Y.name]))
                # check correctness of monotonicities
                # nan column in case no monotonicity is needed
                if mono_fn is None:
                    self.assertTrue(np.isnan(ys['monotonicity']))
                # monotonicities equality in the other case
                else:
                    real_monotonicity = mono_fn(xs, Xs)
                    self.assertEqual(ys['monotonicity'], real_monotonicity)
        except Exception as exception:
            self.fail(exception)

    def test_without_monotonicities(self):
        """Tests the augmentation procedure without including monotonicities computation."""
        self._test(None)

    def test_with_monotonicities(self):
        """Tests the augmentation procedure including monotonicities computation."""

        def _mono_fn(samples: np.ndarray, references: np.ndarray) -> np.ndarray:
            return compute_numeric_monotonicities(samples=samples, references=references, directions=DIRECTIONS)

        self._test(_mono_fn)
