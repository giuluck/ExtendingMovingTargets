import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from moving_targets.metrics import MAE, MSE, R2
from src.datasets import SyntheticManager
from src.models import MTRegressionMaster
from src.util.plot import ColorFader
from test.datasets.managers.test_manager import TestManager, AnalysisCallback


class SyntheticTest(TestManager):
    def __init__(self, extrapolation=False, warm_start=False, **kwargs):
        super(SyntheticTest, self).__init__(
            dataset=SyntheticManager(),
            master_type=MTRegressionMaster,
            metrics=[MAE(), MSE(), R2()],
            data_args=dict(extrapolation=extrapolation),
            augmented_args=dict(num_augmented=15),
            monotonicities_args=dict(kind='group'),
            learner_args=dict(output_act=None, h_units=[16] * 4, optimizer='adam', loss='mse', warm_start=warm_start),
            **kwargs
        )


class SyntheticAdjustments2D(AnalysisCallback):
    label_size = 0.3
    max_size = 100
    alpha = 0.4

    def on_process_start(self, macs, x, y, val_data, **kwargs):
        super(SyntheticAdjustments2D, self).on_process_start(macs, x, y, val_data, **kwargs)
        self.data['ground'] = SyntheticManager.function(self.data['a'], self.data['b'])

    def on_training_end(self, macs, x, y, val_data, iteration, **kwargs):
        self.data[f'pred {iteration}'] = macs.predict(x)
        self.data[f'pred err {iteration}'] = self.data[f'pred {iteration}'] - self.data['ground']

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        self.data[f'adj {iteration}'] = adjusted_y
        self.data[f'adj err {iteration}'] = self.data[f'adj {iteration}'] - self.data['ground']
        self.data[f'sw {iteration}'] = kwargs.get('sample_weight', SyntheticAdjustments2D.label_size * np.ones_like(y))

    def plot_function(self, iteration):
        def synthetic_inverse(column):
            b = np.sin(np.pi * (self.data['b'] - 0.01)) ** 2 + 1
            return (self.data[column] - b) * b

        a, sw, pred = self.data['a'], self.data[f'sw {iteration}'].values, synthetic_inverse(f'pred {iteration}')
        s, m = self.data['mask'].values, dict(aug='o', label='X')
        ls, ms, al = SyntheticAdjustments2D.label_size, SyntheticAdjustments2D.max_size, SyntheticAdjustments2D.alpha
        sns.lineplot(x=self.data['a'], y=synthetic_inverse('ground'), color='green')
        sns.scatterplot(x=a, y=pred, color='red', alpha=al, s=ls * ms / 2)
        if iteration == AnalysisCallback.PRETRAINING:
            adj, color = synthetic_inverse('label'), 'black'
        else:
            adj, color = synthetic_inverse(f'adj {iteration}'), 'blue'
        # rescale in case of uniform values
        if np.allclose(sw, 1.0):
            sw *= ls
        else:
            sw[s == 'label'] = ls
        sns.scatterplot(x=a, y=adj, style=s, markers=m, size=sw, size_norm=(0, 1), sizes=(0, ms), color=color, alpha=al)
        plt.legend(['ground', 'predictions', 'labels' if iteration == AnalysisCallback.PRETRAINING else 'adjusted'])


class SyntheticAdjustments3D(AnalysisCallback):
    label_size = 0.3
    max_size = 100

    def __init__(self, res=100, **kwargs):
        super(SyntheticAdjustments3D, self).__init__(**kwargs)
        assert self.sorting_attribute is None, 'sorting_attribute must be None'
        self.res = res
        self.val = None

    def on_process_start(self, macs, x, y, val_data, **kwargs):
        super(SyntheticAdjustments3D, self).on_process_start(macs, x, y, val_data, **kwargs)
        # swap values and data in order to print the grid
        self.val = self.data.copy()
        a, b = np.meshgrid(np.linspace(-1, 1, self.res), np.linspace(-1, 1, self.res))
        self.data = pd.DataFrame.from_dict({'a': a.flatten(), 'b': b.flatten()})

    def on_training_end(self, macs, x, y, val_data, iteration, **kwargs):
        self.val[f'pred {iteration}'] = macs.predict(x)
        self.data[f'z {iteration}'] = macs.predict(self.data[['a', 'b']])

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        self.val[f'adj {iteration}'] = adjusted_y
        self.val[f'sw {iteration}'] = kwargs.get('sample_weight', SyntheticAdjustments3D.label_size * np.ones_like(y))

    def plot_function(self, iteration):
        # plot 3D response
        ga = self.data['a'].values.reshape(self.res, self.res)
        gb = self.data['b'].values.reshape(self.res, self.res)
        gz = self.data[f'z {iteration}'].values.reshape(self.res, self.res)
        plt.pcolor(ga, gb, gz, shading='auto', cmap='viridis', vmin=gz.min(), vmax=gz.max())
        # plot sample weights
        m, s = self.val['mask'].values == 'aug', (0, SyntheticAdjustments3D.max_size)
        a, b, pred, sw = self.val['a'], self.val['b'], self.val[f'pred {iteration}'], self.val[f'sw {iteration}'].values
        ls = SyntheticAdjustments3D.label_size * SyntheticAdjustments3D.max_size
        sns.scatterplot(x=a[~m], y=b[~m], s=ls, size_norm=(0, 1), sizes=s, color='black', marker='X', legend=False)
        if iteration != AnalysisCallback.PRETRAINING:
            sw = sw[m] * SyntheticAdjustments3D.label_size if np.allclose(sw, 1.0) else sw[m]
            sns.scatterplot(x=a[m], y=b[m], size=sw, size_norm=(0, 1), sizes=s, color='black', marker='o', legend=False)
        plt.legend(['ground', 'label', 'adjusted'])


class SyntheticResponse(AnalysisCallback):
    def __init__(self, res=10, **kwargs):
        super(SyntheticResponse, self).__init__(**kwargs)
        a, b = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
        self.grid = pd.DataFrame.from_dict({'a': a.flatten(), 'b': b.flatten()})
        self.fader = ColorFader('red', 'blue', bounds=(-1, 1))

    def on_training_end(self, macs, x, y, val_data, iteration, **kwargs):
        input_grid = self.grid[['a', 'b']]
        self.grid[f'pred {iteration}'] = macs.predict(input_grid)

    def plot_function(self, iteration):
        for idx, group in self.grid.groupby('b'):
            label = f'b = {idx:.0f}' if idx in [-1, 1] else None
            sns.lineplot(data=group, x='a', y=f'pred {iteration}', color=self.fader(idx), alpha=0.4, label=label)
