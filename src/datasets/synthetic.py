"""Synthetic Data Manager."""

from typing import Any, Dict, Optional, Callable, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.datasets.manager import Manager, AnalysisCallback
from src.util.analysis import ColorFader


class Synthetic(Manager):
    """Data Manager for the Synthetic Dataset."""

    __name__: str = 'synthetic'

    callbacks: Dict[str, Callable] = {
        **Manager.callbacks,
        'response': lambda fs: SyntheticResponse(file_signature=fs),
        'adjustments_2D': lambda fs: SyntheticAdjustments2D(file_signature=fs),
        'adjustments_3D': lambda fs: SyntheticAdjustments3D(file_signature=fs)
    }

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

    @classmethod
    def load(cls) -> Dict[str, pd.DataFrame]:
        return {'train': Synthetic.sample(n=200, testing_set=False), 'test': Synthetic.sample(n=500, testing_set=True)}

    @classmethod
    def grid(cls, plot: bool = True) -> pd.DataFrame:
        res = 50 if plot else 80
        a, b = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
        return pd.DataFrame({'a': a.flatten(), 'b': b.flatten()})

    def __init__(self):
        super(Synthetic, self).__init__(label='y', directions={'a': 1}, classification=False)

    def _plot(self, model):
        # get data
        grid = self.grid(plot=True)
        grid['pred'] = model.predict(grid)
        grid['label'] = Synthetic.function(grid['a'], grid['b'])
        res = np.sqrt(len(grid)).astype(int)
        fader = ColorFader('red', 'blue', bounds=[-1, 1])
        _, axes = plt.subplots(2, 3, figsize=(16, 9), tight_layout=True)
        for ax, (title, y) in zip(axes, {'Ground Truth': 'label', 'Estimated Function': 'pred'}.items()):
            # plot bivariate function
            ax[0].pcolor(grid['a'].values.reshape((res, res)),
                         grid['b'].values.reshape((res, res)),
                         grid[y].values.reshape((res, res)),
                         vmin=grid['label'].min(),
                         vmax=grid['label'].max(),
                         shading='auto',
                         cmap='viridis')
            # plot first feature (with title as it is the central plot)
            for idx, group in grid.groupby('b'):
                label = f'b = {idx:.0f}' if idx in [-1, 1] else None
                sns.lineplot(data=group, x='a', y=y, color=fader(idx), alpha=0.4, label=label, ax=ax[1])
            # plot second feature
            for idx, group in grid.groupby('a'):
                label = f'a = {idx:.0f}' if idx in [-1, 1] else None
                sns.lineplot(data=group, x='b', y=y, color=fader(idx), alpha=0.4, label=label, ax=ax[2])


class SyntheticResponse(AnalysisCallback):
    """Investigates marginal feature responses during iterations in synthetic dataset.

    - 'res' is the meshgrid resolution.
    """

    def __init__(self,
                 res: int = 10,
                 sorting_attribute: Optional[str] = None,
                 file_signature: Optional[str] = None,
                 num_columns: Union[int, str] = 'auto'):
        super(SyntheticResponse, self).__init__(sorting_attribute=sorting_attribute,
                                                file_signature=file_signature,
                                                num_columns=num_columns)
        a, b = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
        self.data: pd.DataFrame = pd.DataFrame.from_dict({'a': a.flatten(), 'b': b.flatten()})
        self.fader: ColorFader = ColorFader('red', 'blue', bounds=[-1, 1])

    def on_training_end(self, macs, x, y: np.ndarray, val_data: Optional[Manager]):
        self.data[f'pred {macs.iteration}'] = macs.predict(self.data[['a', 'b']])

    def _plot_function(self, iteration: int) -> Optional[str]:
        for idx, group in self.data.groupby('b'):
            label = f'b = {idx:.0f}' if idx in [-1, 1] else None
            sns.lineplot(data=group, x='a', y=f'pred {iteration}', color=self.fader(idx), alpha=0.4, label=label)
        return


class SyntheticAdjustments2D(AnalysisCallback):
    """Investigates Moving Targets adjustments (on the monotonic feature) during iterations in synthetic dataset."""

    def on_process_start(self, macs, x, y, val_data):
        self.data = x.reset_index(drop=True)
        self.data['y'] = pd.Series(y, name='y')
        self.data['ground'] = Synthetic.function(x['a'], x['b'])

    def on_training_end(self, macs, x, y, val_data):
        predictions = macs.predict(x)
        self.data[f'pred {macs.iteration}'] = predictions
        self.data[f'pred err {macs.iteration}'] = predictions - self.data['ground']

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data):
        self.data[f'adj {macs.iteration}'] = adjusted_y
        self.data[f'adj err {macs.iteration}'] = adjusted_y - self.data['ground']

    def _plot_function(self, iteration: int) -> Optional[str]:
        def synthetic_inverse(column):
            """Computes the value of the expected value of the 'a' feature given the output label."""
            b = np.sin(np.pi * (self.data['b'] - 0.01)) ** 2 + 1
            return (self.data[column] - b) * b

        a, pred, alpha = self.data['a'], synthetic_inverse(f'pred {iteration}'), 0.4
        sns.lineplot(x=self.data['a'], y=synthetic_inverse('ground'), color='green')
        sns.scatterplot(x=a, y=pred, color='red', alpha=alpha)
        if iteration == 0:
            sns.scatterplot(x=a, y=synthetic_inverse('y'), color='black', alpha=alpha)
            plt.legend(['ground', 'predictions', 'labels'])
        else:
            sns.scatterplot(x=a, y=synthetic_inverse(f'adj {iteration}'), color='blue', alpha=alpha)
            plt.legend(['ground', 'predictions', 'adjusted'])
        return


class SyntheticAdjustments3D(AnalysisCallback):
    """Investigates Moving Targets adjustments (on both the features) during iterations in synthetic dataset.

    - 'res' is the meshgrid resolution.
    """

    def __init__(self,
                 res: int = 100,
                 file_signature: Optional[str] = None,
                 num_columns: Union[int, str] = 'auto'):
        super(SyntheticAdjustments3D, self).__init__(sorting_attribute=None,
                                                     file_signature=file_signature,
                                                     num_columns=num_columns)
        a, b = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
        self.data: pd.DataFrame = pd.DataFrame.from_dict({'a': a.flatten(), 'b': b.flatten()})
        self.data['ground'] = Synthetic.function(self.data['a'], self.data['b'])

    def on_training_end(self, macs, x, y: np.ndarray, val_data: Optional[Manager]):
        self.data[f'z {macs.iteration}'] = macs.predict(self.data[['a', 'b']])

    def _plot_function(self, iteration: int) -> Optional[str]:
        # plot 3D response
        res = np.sqrt(len(self.data)).astype(int)
        ga = self.data['a'].values.reshape(res, res)
        gb = self.data['b'].values.reshape(res, res)
        gz = self.data[f'z {iteration}'].values.reshape(res, res)
        gg = self.data['ground']
        plt.pcolor(ga, gb, gz, shading='auto', cmap='viridis', vmin=gg.min(), vmax=gg.max())
        return
