"""Synthetic Data Manager."""

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.datasets.dataset import Dataset


class Synthetic(Dataset):
    """Data Manager for the Synthetic Dataset."""

    @staticmethod
    def function(a, b) -> Any:
        """Ground function."""
        a = a ** 3
        b = np.sin(np.pi * (b - 0.01)) ** 2 + 1
        return a / b + b

    @staticmethod
    def sample(n: int, testing_set: bool = True) -> pd.DataFrame:
        """Sample data points and computes the respective function value. Depending on the value of 'testing_set',
        samples either from the train or test distribution."""
        rng = np.random.default_rng(seed=0)
        a = rng.uniform(low=-1, high=1, size=n) if testing_set else rng.normal(scale=0.3, size=n).clip(min=-1, max=1)
        b = rng.uniform(low=-1, high=1, size=n)
        y = Synthetic.function(a, b)
        return pd.DataFrame.from_dict({'a': a, 'b': b, 'y': y})

    @staticmethod
    def load() -> Dict[str, pd.DataFrame]:
        return {'train': Synthetic.sample(n=200, testing_set=False), 'test': Synthetic.sample(n=500, testing_set=True)}

    def __init__(self):
        a, b = np.meshgrid(np.linspace(-1, 1, 80), np.linspace(-1, 1, 80))
        super(Synthetic, self).__init__(label='y',
                                        directions={'a': 1},
                                        classification=False,
                                        grid=pd.DataFrame({'a': a.flatten(), 'b': b.flatten()}))
