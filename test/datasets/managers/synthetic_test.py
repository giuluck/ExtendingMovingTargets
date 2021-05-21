from typing import Any, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.datasets import SyntheticManager
from src.util.plot import ColorFader
from test.datasets.managers.test_manager import RegressionTest, AnalysisCallback


class SyntheticTest(RegressionTest):
    def __init__(self, noise: float = 0.0, **kwargs):
        super(SyntheticTest, self).__init__(
            dataset=SyntheticManager(noise=noise),
            augmented_args=dict(num_augmented=15),
            monotonicities_args=dict(kind='group'),
            **kwargs
        )


class SyntheticAdjustments2D(AnalysisCallback):
    max_size = 30
    alpha = 0.4

    def on_process_start(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], **kwargs):
        super(SyntheticAdjustments2D, self).on_process_start(macs, x, y, val_data, **kwargs)
        self.data['ground'] = SyntheticManager.function(self.data['a'], self.data['b'])

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration, **kwargs):
        self.data[f'pred {iteration}'] = macs.predict(x)
        self.data[f'pred err {iteration}'] = self.data[f'pred {iteration}'] - self.data['ground']

    def on_adjustment_end(self, macs, x: Matrix, y: Vector, adjusted_y: Vector, val_data: Optional[Dataset], iteration: Iteration, **kwargs):
        self.data[f'adj {iteration}'] = adjusted_y
        self.data[f'adj err {iteration}'] = self.data[f'adj {iteration}'] - self.data['ground']
        self.data[f'sw {iteration}'] = kwargs.get('sample_weight', np.where(self.data['mask'] == 'label', 1, 0))

    def plot_function(self, iteration: Iteration) -> Optional[str]:
        def synthetic_inverse(column):
            b = np.sin(np.pi * (self.data['b'] - 0.01)) ** 2 + 1
            return (self.data[column] - b) * b

        a, sw, pred = self.data['a'], self.data[f'sw {iteration}'].values, synthetic_inverse(f'pred {iteration}')
        s, m = self.data['mask'].values, AnalysisCallback.MARKERS
        ms, al = SyntheticAdjustments2D.max_size, SyntheticAdjustments2D.alpha
        sns.lineplot(x=self.data['a'], y=synthetic_inverse('ground'), color='green')
        sns.scatterplot(x=a, y=pred, color='red', alpha=al, s=ms)
        if iteration == AnalysisCallback.PRETRAINING:
            adj, color = synthetic_inverse('label'), 'black'
        else:
            adj, color = synthetic_inverse(f'adj {iteration}'), 'blue'
        sns.scatterplot(x=a, y=adj, style=s, markers=m, size=sw, size_norm=(0, 1), sizes=(0, ms), color=color, alpha=al)
        plt.legend(['ground', 'predictions', 'labels' if iteration == AnalysisCallback.PRETRAINING else 'adjusted'])
        return


class SyntheticAdjustments3D(AnalysisCallback):
    max_size = 40

    def __init__(self, res: int = 100, data_points: bool = True, **kwargs):
        super(SyntheticAdjustments3D, self).__init__(**kwargs)
        assert self.sorting_attribute is None, 'sorting_attribute must be None'
        self.res: int = res
        self.data_points: bool = data_points
        self.val: Optional[pd.DataFrame] = None

    def on_process_start(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], **kwargs):
        super(SyntheticAdjustments3D, self).on_process_start(macs, x, y, val_data, **kwargs)
        # swap values and data in order to print the grid
        self.val = self.data.copy()
        a, b = np.meshgrid(np.linspace(-1, 1, self.res), np.linspace(-1, 1, self.res))
        self.data = pd.DataFrame.from_dict({'a': a.flatten(), 'b': b.flatten()})

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration, **kwargs):
        self.val[f'pred {iteration}'] = macs.predict(x)
        self.data[f'z {iteration}'] = macs.predict(self.data[['a', 'b']])

    def on_adjustment_end(self, macs, x: Matrix, y: Vector, adjusted_y: Vector, val_data: Optional[Dataset], iteration: Iteration, **kwargs):
        self.val[f'adj {iteration}'] = adjusted_y
        self.val[f'sw {iteration}'] = kwargs.get('sample_weight', np.where(self.val['mask'] == 'label', 1, 0))

    def plot_function(self, iteration: Iteration) -> Optional[str]:
        # plot 3D response
        ga = self.data['a'].values.reshape(self.res, self.res)
        gb = self.data['b'].values.reshape(self.res, self.res)
        gz = self.data[f'z {iteration}'].values.reshape(self.res, self.res)
        plt.pcolor(ga, gb, gz, shading='auto', cmap='viridis', vmin=gz.min(), vmax=gz.max())
        # plot data points
        if self.data_points:
            markers, sizes = AnalysisCallback.MARKERS, (0, SyntheticAdjustments3D.max_size)
            sns.scatterplot(data=self.val, x='a', y='b', size=f'sw {iteration}', size_norm=(0, 1), sizes=sizes,
                            color='black', style='mask', markers=markers, legend=False)
        plt.legend(['ground', 'label', 'adjusted'])
        return


class SyntheticResponse(AnalysisCallback):
    def __init__(self, res: int = 10, **kwargs):
        super(SyntheticResponse, self).__init__(**kwargs)
        a, b = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
        self.grid = pd.DataFrame.from_dict({'a': a.flatten(), 'b': b.flatten()})
        self.fader = ColorFader('red', 'blue', bounds=(-1, 1))

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration, **kwargs):
        input_grid = self.grid[['a', 'b']]
        self.grid[f'pred {iteration}'] = macs.predict(input_grid)

    def plot_function(self, iteration: Iteration) -> Optional[str]:
        for idx, group in self.grid.groupby('b'):
            label = f'b = {idx:.0f}' if idx in [-1, 1] else None
            sns.lineplot(data=group, x='a', y=f'pred {iteration}', color=self.fader(idx), alpha=0.4, label=label)
        return
