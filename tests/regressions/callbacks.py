import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.regressions import synthetic_function
from src.util.plot import ColorFader
from tests.util.callbacks import AnalysisCallback, PRETRAINING


class SyntheticAdjustments2D(AnalysisCallback):
    label_size = 0.3
    max_size = 100
    alpha = 0.4

    def on_process_start(self, macs, x, y, val_data, **kwargs):
        super(SyntheticAdjustments2D, self).on_process_start(macs, x, y, val_data, **kwargs)
        self.data['ground'] = synthetic_function(self.data['a'], self.data['b'])

    def on_training_end(self, macs, x, y, val_data, iteration, **kwargs):
        self.data[f'pred {iteration}'] = self.y_scaler.invert(macs.predict(x))
        self.data[f'pred err {iteration}'] = self.data[f'pred {iteration}'] - self.data['ground']

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        self.data[f'adj {iteration}'] = self.y_scaler.invert(adjusted_y)
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
        if iteration == PRETRAINING:
            adj, color = synthetic_inverse('label'), 'black'
        else:
            adj, color = synthetic_inverse(f'adj {iteration}'), 'blue'
        # rescale in case of uniform values
        if np.allclose(sw, 1.0):
            sw *= ls
        else:
            sw[s == 'label'] = ls
        sns.scatterplot(x=a, y=adj, style=s, markers=m, size=sw, size_norm=(0, 1), sizes=(0, ms), color=color, alpha=al)
        plt.legend(['ground', 'predictions', 'labels' if iteration == PRETRAINING else 'adjusted'])


class SyntheticAdjustments3D(AnalysisCallback):
    label_size = 0.3
    max_size = 100

    def __init__(self, scalers, res=100, **kwargs):
        super(SyntheticAdjustments3D, self).__init__(scalers=scalers, **kwargs)
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
        self.val[f'pred {iteration}'] = self.y_scaler.invert(macs.predict(x))
        self.data[f'z {iteration}'] = self.y_scaler.invert(macs.predict(self.x_scaler.transform(self.data[['a', 'b']])))

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        self.val[f'adj {iteration}'] = self.y_scaler.invert(adjusted_y)
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
        if iteration != PRETRAINING:
            sw = sw[m] * SyntheticAdjustments3D.label_size if np.allclose(sw, 1.0) else sw[m]
            sns.scatterplot(x=a[m], y=b[m], size=sw, size_norm=(0, 1), sizes=s, color='black', marker='o', legend=False)
        plt.legend(['ground', 'label', 'adjusted'])


class SyntheticResponse(AnalysisCallback):
    def __init__(self, scalers, res=10, **kwargs):
        super(SyntheticResponse, self).__init__(scalers=scalers, **kwargs)
        a, b = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
        self.grid = pd.DataFrame.from_dict({'a': a.flatten(), 'b': b.flatten()})
        self.fader = ColorFader('red', 'blue', bounds=(-1, 1))

    def on_training_end(self, macs, x, y, val_data, iteration, **kwargs):
        input_grid = self.x_scaler.transform(self.grid[['a', 'b']])
        self.grid[f'pred {iteration}'] = self.y_scaler.invert(macs.predict(input_grid))

    def plot_function(self, iteration):
        for idx, group in self.grid.groupby('b'):
            label = f'b = {idx:.0f}' if idx in [-1, 1] else None
            sns.lineplot(data=group, x='a', y=f'pred {iteration}', color=self.fader(idx), alpha=0.4, label=label)


class CarsAdjustments(AnalysisCallback):
    label_size = 0.4
    max_size = 100
    alpha = 0.4

    def __init__(self, scalers, plot_kind='scatter', **kwargs):
        super(CarsAdjustments, self).__init__(scalers=scalers, **kwargs)
        assert plot_kind in ['line', 'scatter'], "plot_kind should be either 'line' or 'scatter'"
        self.plot_kind = plot_kind

    def on_training_end(self, macs, x, y, val_data, iteration, **kwargs):
        self.data[f'pred {iteration}'] = self.y_scaler.invert(macs.predict(x))

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        self.data[f'adj {iteration}'] = self.y_scaler.invert(adjusted_y)
        self.data[f'sw {iteration}'] = kwargs.get('sample_weight', CarsAdjustments.label_size * np.ones_like(y))

    def plot_function(self, iteration):
        x, y = self.data['price'].values, self.data['sales'].values
        s, m = np.array(self.data['mask']), dict(aug='o', label='X')
        sn, al = (0, CarsAdjustments.max_size), CarsAdjustments.alpha
        p, adj, sw = self.data[f'pred {iteration}'], self.data[f'adj {iteration}'], self.data[f'sw {iteration}'].values
        # rescale in case of uniform values
        if np.allclose(sw, 1.0):
            sw *= CarsAdjustments.label_size
        else:
            sw[s == 'label'] = CarsAdjustments.label_size
        if iteration == PRETRAINING:
            sns.scatterplot(x=x, y=y, style=s, markers=m, size=sw, size_norm=(0, 1), sizes=sn, color='black', alpha=al)
        elif self.plot_kind == 'line':
            sns.lineplot(x=x, y=adj, color='blue')
            for i in range(self.data.shape[0]):
                plt.plot([x[i], x[i]], [y[i], adj[i]], c='black', alpha=al)
        elif self.plot_kind == 'scatter':
            sns.scatterplot(x=x, y=adj, style=s, markers=m, size=sw, size_norm=(0, 1), sizes=sn, color='blue', alpha=al)
        sns.lineplot(x=x, y=p, color='red')
        plt.legend(['predictions', 'labels' if iteration == PRETRAINING else 'adjusted'])
        return f'{iteration}) adj. mae = {np.abs((adj - y).fillna(0)).mean():.4f}'


class PuzzlesResponse(AnalysisCallback):
    features = ['word_count', 'star_rating', 'num_reviews']

    def __init__(self, scalers, feature, res=5, **kwargs):
        super(PuzzlesResponse, self).__init__(scalers=scalers, **kwargs)
        assert feature in self.features, f"feature should be in {self.features}"
        grid = np.meshgrid(np.linspace(0, 1, res), np.linspace(0, 1, res), np.linspace(0, 1, res))
        self.grid = self.x_scaler.invert(pd.DataFrame.from_dict({k: v.flatten() for k, v in zip(self.features, grid)}))
        self.feature = feature

    def on_training_end(self, macs, x, y, val_data, iteration, **kwargs):
        input_grid = self.x_scaler.transform(self.grid[self.features])
        self.grid[f'pred {iteration}'] = self.y_scaler.invert(macs.predict(input_grid))

    def plot_function(self, iteration):
        fi, fj = [f for f in self.features if f != self.feature]
        li, ui = self.grid[fi].min(), self.grid[fi].max()
        lj, uj = self.grid[fj].min(), self.grid[fj].max()
        fader = ColorFader('black', 'magenta', 'cyan', 'yellow', bounds=(li, lj, ui, uj))
        for (i, j), group in self.grid.groupby([fi, fj]):
            label = f'{fi}: {i:.0f}, {fj}: {j:.0f}' if (i in [li, ui] and j in [lj, uj]) else None
            sns.lineplot(data=group, x=self.feature, y=f'pred {iteration}', color=fader(i, j), alpha=0.6, label=label)
        return f'{iteration}) {self.feature.replace("_", " ").upper()}'
