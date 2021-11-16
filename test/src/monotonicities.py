"""Monotonicities Tests."""
import unittest
from collections import Callable

import numpy as np
import pandas as pd

from src.datasets import RestaurantsManager
from src.util.augmentation import compute_numeric_monotonicities


class TestMonotonicities(unittest.TestCase):
    """Tests the correctness of the monotonicities computation."""

    def _test(self, data_points: np.ndarray, monotonicities: np.ndarray, monotonicities_fn: Callable):
        """Performs the tests.

        :data_points:
            The input data.

        :param monotonicities:
            The matrix of expected monotonicities.

        :param monotonicities_fn:
            A callable function to compute monotonicities.
        """
        try:
            # check 1 vs 1 monotonicities (each single data point with one single other data point as reference)
            for i, sample in enumerate(data_points):
                for j, reference in enumerate(data_points):
                    expected = monotonicities[i, j]
                    computed = monotonicities_fn(samples=sample, references=reference)
                    self.assertIsInstance(computed, np.int32, msg=f'sample={i}, reference={j}')
                    self.assertEqual(computed, expected, msg=f'sample={i}, reference={j}')
            # check 1 vs many monotonicities (each single data point with all the other data points as references)
            for i, sample in enumerate(data_points):
                expected = monotonicities[i, :]
                computed = monotonicities_fn(samples=sample, references=data_points[:])
                self.assertIsInstance(computed, np.ndarray, msg=f'sample={i}, references=all')
                self.assertEqual(expected.shape, computed.shape, msg=f'sample={i}, references=all')
                self.assertListEqual(list(expected), list(computed), msg=f'sample={i}, references=all')
            # check many vs 1 monotonicities (all the data points with one single other data point as reference)
            for j, reference in enumerate(data_points):
                expected = monotonicities[:, j].flatten()
                computed = monotonicities_fn(samples=data_points[:], references=reference)
                self.assertIsInstance(computed, np.ndarray, msg=f'samples=all, reference={j}')
                self.assertEqual(expected.shape, computed.shape, msg=f'samples=all, reference={j}')
                self.assertListEqual(list(expected), list(computed), msg=f'samples=all, reference={j}')
            # check many vs many monotonicities (all the data points with all the other data points as references)
            expected = monotonicities[:, :]
            computed = monotonicities_fn(samples=data_points[:], references=data_points[:])
            self.assertIsInstance(computed, np.ndarray, msg='samples=all, references=all')
            self.assertEqual(expected.shape, computed.shape, msg='samples=all, references=all')
            expected, computed = list([list(v) for v in expected]), list([list(v) for v in computed])
            self.assertListEqual(expected, computed, msg='samples=all, references=all')
        except Exception as exception:
            self.fail(exception)

    def test_univariate(self):
        """Tests the computation of univariate monotonicities."""

        def _mono_fn(samples: np.ndarray, references: np.ndarray) -> np.ndarray:
            return compute_numeric_monotonicities(samples=samples, references=references, directions=np.array([1]))

        data = np.array([[0], [1], [2], [3]])
        mono = np.array([
            [.0, -1, -1, -1],
            [+1, .0, -1, -1],
            [+1, +1, .0, -1],
            [+1, +1, +1, .0]
        ]).astype(int)
        self._test(data_points=data, monotonicities=mono, monotonicities_fn=_mono_fn)

    def test_numeric(self):
        """Tests the computation of multivariate numeric monotonicities."""

        def _mono_fn(samples: np.ndarray, references: np.ndarray) -> np.ndarray:
            directions = np.array([1, 0, -1])
            return compute_numeric_monotonicities(samples=samples, references=references, directions=directions)

        data = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ])
        mono = np.array([
            [.0, -1, .0, +1, .0, .0, .0, .0],
            [+1, .0, .0, .0, .0, +1, .0, .0],
            [.0, .0, .0, .0, -1, .0, +1, .0],
            [-1, .0, .0, .0, .0, -1, .0, .0],
            [.0, .0, +1, .0, .0, .0, .0, +1],
            [.0, -1, .0, +1, .0, .0, .0, .0],
            [.0, .0, -1, .0, .0, .0, .0, -1],
            [.0, .0, .0, .0, -1, .0, +1, .0]
        ]).astype(int)
        self._test(data_points=data, monotonicities=mono, monotonicities_fn=_mono_fn)

    def test_restaurants(self):
        """Tests the computation of multivariate categorical monotonicities, as in the restaurants dataset."""

        def _mono_fn(samples: np.ndarray, references: np.ndarray) -> np.ndarray:
            columns = ['avg_rating', 'num_reviews', 'D', 'DD', 'DDD', 'DDDD']
            samples = pd.DataFrame(samples.reshape(-1, 6), columns=columns)
            if references.ndim == 1:
                references = pd.DataFrame(references.reshape(-1, 6), columns=columns).iloc[0]
            else:
                references = pd.DataFrame(references.reshape(-1, 6), columns=columns)
            # noinspection PyTypeChecker
            return RestaurantsManager.compute_monotonicities(self=None, samples=samples, references=references)

        data = np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0]
        ])
        mono = np.array([
            [.0, -1, .0, .0, -1, -1, .0],
            [+1, .0, .0, +1, .0, .0, .0],
            [.0, .0, .0, .0, .0, .0, .0],
            [.0, -1, .0, .0, .0, .0, .0],
            [+1, .0, .0, .0, .0, .0, -1],
            [+1, .0, .0, .0, .0, .0, -1],
            [.0, .0, .0, .0, +1, +1, .0]
        ]).astype(int)

        self._test(data_points=data, monotonicities=mono, monotonicities_fn=_mono_fn)
