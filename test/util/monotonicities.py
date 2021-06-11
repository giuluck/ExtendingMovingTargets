"""Monotonicities Tests."""
import unittest
from collections import Callable

import numpy as np
import pandas as pd

from src.datasets import RestaurantsManager
from src.util.augmentation import compute_numeric_monotonicities


class TestMonotonicities(unittest.TestCase):
    def _test(self, data_points: np.ndarray, monotonicities: np.ndarray, monotonicities_fn: Callable):
        for i, sample in enumerate(data_points):
            for j, reference in enumerate(data_points):
                # check each single data point with one single other data point as reference
                expected = monotonicities[i, j]
                computed = monotonicities_fn(samples=sample, references=reference)
                self.assertIsInstance(computed, np.int32, msg=f'sample={i}, reference={j}')
                self.assertEqual(computed, expected, msg=f'sample={i}, reference={j}')
        for i, sample in enumerate(data_points):
            # check each single data point with all the other data points as references
            expected = monotonicities[i, :]
            computed = monotonicities_fn(samples=sample, references=data_points[:])
            self.assertIsInstance(computed, np.ndarray, msg=f'sample={i}, references=all')
            self.assertEqual(expected.shape, computed.shape, msg=f'sample={i}, references=all')
            self.assertListEqual(list(expected), list(computed), msg=f'sample={i}, references=all')
        for j, reference in enumerate(data_points):
            # check all the data points with one single other data point as reference
            expected = monotonicities[:, j].flatten()
            computed = monotonicities_fn(samples=data_points[:], references=reference)
            self.assertIsInstance(computed, np.ndarray, msg=f'samples=all, reference={j}')
            self.assertEqual(expected.shape, computed.shape, msg=f'samples=all, reference={j}')
            self.assertListEqual(list(expected), list(computed), msg=f'samples=all, reference={j}')
        # check all the data points with one with all the other data points as references
        expected = monotonicities[:, :]
        computed = monotonicities_fn(samples=data_points[:], references=data_points[:])
        self.assertIsInstance(computed, np.ndarray, msg='samples=all, references=all')
        self.assertEqual(expected.shape, computed.shape, msg='samples=all, references=all')
        expected, computed = list([list(v) for v in expected]), list([list(v) for v in computed])
        self.assertListEqual(expected, computed, msg='samples=all, references=all')

    def test_univariate(self):
        data = np.array([[0], [1], [2], [3]])
        mono = np.array([
            [.0, -1, -1, -1],
            [+1, .0, -1, -1],
            [+1, +1, .0, -1],
            [+1, +1, +1, .0]
        ]).astype(int)
        mono_fn = lambda samples, references: compute_numeric_monotonicities(samples=samples,
                                                                             references=references,
                                                                             directions=np.array([1]))
        self._test(data_points=data, monotonicities=mono, monotonicities_fn=mono_fn)

    def test_numeric(self):
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
        mono_fn = lambda samples, references: compute_numeric_monotonicities(samples=samples,
                                                                             references=references,
                                                                             directions=np.array([1, 0, -1]))
        self._test(data_points=data, monotonicities=mono, monotonicities_fn=mono_fn)

    def test_restaurants(self):
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

        def _mono_fn(samples: np.ndarray, references: np.ndarray) -> np.ndarray:
            columns = ['avg_rating', 'num_reviews', 'D', 'DD', 'DDD', 'DDDD']
            samples = pd.DataFrame(samples.reshape(-1, 6), columns=columns)
            if references.ndim == 1:
                references = pd.DataFrame(references.reshape(-1, 6), columns=columns).iloc[0]
            else:
                references = pd.DataFrame(references.reshape(-1, 6), columns=columns)
            # noinspection PyTypeChecker
            return RestaurantsManager.compute_monotonicities(self=None, samples=samples, references=references)

        self._test(data_points=data, monotonicities=mono, monotonicities_fn=_mono_fn)
