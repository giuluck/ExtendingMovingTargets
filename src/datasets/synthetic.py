"""Synthetic Data Manager."""

from typing import Any, Dict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.datasets.dataset import Dataset
from src.util.analysis import ColorFader


class Synthetic(Dataset):
    """Data Manager for the Synthetic Dataset."""

    __name__: str = 'synthetic'

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

    def _plot(self, model):
        # get data
        grid = self.grid.copy()
        grid['pred'] = model.predict(grid)
        grid['label'] = Synthetic.function(grid['a'], grid['b'])
        fader = ColorFader('red', 'blue', bounds=[-1, 1])
        _, axes = plt.subplots(2, 3, figsize=(16, 9), tight_layout=True)
        for ax, (title, y) in zip(axes, {'Ground Truth': 'label', 'Estimated Function': 'pred'}.items()):
            # plot bivariate function
            ax[0].pcolor(grid['a'].values.reshape(80, 80), grid['b'].values.reshape(80, 80),
                         grid[y].values.reshape(80, 80), shading='auto', cmap='viridis',
                         vmin=grid['label'].min(), vmax=grid['label'].max())
            # plot first feature (with title as it is the central plot)
            for idx, group in grid.groupby('b'):
                label = f'b = {idx:.0f}' if idx in [-1, 1] else None
                sns.lineplot(data=group, x='a', y=y, color=fader(idx), alpha=0.4, label=label, ax=ax[1])
            # plot second feature
            for idx, group in grid.groupby('a'):
                label = f'a = {idx:.0f}' if idx in [-1, 1] else None
                sns.lineplot(data=group, x='b', y=y, color=fader(idx), alpha=0.4, label=label, ax=ax[2])
