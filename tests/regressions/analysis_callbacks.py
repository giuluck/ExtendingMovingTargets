import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from moving_targets.callbacks import Callback
from src.regressions import synthetic_function
from src.util.plot import ColorFader

PRETRAINING = 'PT'


class AnalysisCallback(Callback):
    def __init__(self, scalers, num_columns=5, sorting_attributes=None, file_signature=None, do_plot=True, **kwargs):
        super(AnalysisCallback, self).__init__()
        self.x_scaler = scalers[0]
        self.y_scaler = scalers[1]
        self.num_columns = num_columns
        self.sorting_attributes = sorting_attributes
        self.file_signature = file_signature
        self.do_plot = do_plot
        self.plot_kwargs = {'figsize': (20, 10), 'tight_layout': True}
        self.plot_kwargs.update(kwargs)
        self.data = None
        self.iterations = []

    def on_process_start(self, macs, x, y, val_data):
        x = self.x_scaler.invert(x)
        y = self.y_scaler.invert(y)
        self.data = pd.concat((x, y), axis=1)

    def on_pretraining_end(self, macs, x, y, val_data):
        self.on_training_end(macs, x, y, val_data, PRETRAINING)
        self.on_adjustment_end(macs, x, y, np.ones_like(y) * np.nan, val_data, PRETRAINING, )
        self.on_iteration_end(macs, x, y, val_data, PRETRAINING)

    def on_iteration_end(self, macs, x, y, val_data, iteration):
        self.iterations.append(iteration)

    def on_process_end(self, macs, x, y, val_data):
        # sort values
        if self.sorting_attributes is not None:
            self.data = self.data.sort_values(self.sorting_attributes)
        # write on files
        if self.file_signature is not None:
            self.data.to_csv(self.file_signature + '.csv', index_label='index')
            with open(self.file_signature + '.txt', 'w') as f:
                f.write(str(self.data))
        # do plots
        if self.do_plot:
            plt.figure(**self.plot_kwargs)
            num_rows = int(np.ceil(len(self.iterations) / self.num_columns))
            ax = None
            for idx, it in enumerate(self.iterations):
                ax = plt.subplot(num_rows, self.num_columns, idx + 1, sharex=ax, sharey=ax)
                title = self.plot_function(it)
                ax.set(xlabel='', ylabel='')
                ax.set_title(f'{it})' if title is None else title)
            plt.show()

    def plot_function(self, iteration):
        pass


class GroundAnalysis(AnalysisCallback):
    def __init__(self, scalers, num_columns=1, **kwargs):
        super(GroundAnalysis, self).__init__(scalers=scalers, num_columns=num_columns, **kwargs)

    def on_pretraining_end(self, macs, x, y, val_data):
        pass

    def on_training_end(self, macs, x, y, val_data, iteration):
        self.data[f'pred {iteration}'] = self.y_scaler.invert(macs.predict(x))

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        self.data[f'adj {iteration}'] = self.y_scaler.invert(adjusted_y)

    def on_process_end(self, macs, x, y, val_data):
        self.data = self.data[~np.isnan(y)]
        super(GroundAnalysis, self).on_process_end(macs, x, y, val_data)

    def plot_function(self, iteration):
        x = np.arange(len(self.data))
        pred, adj = self.data[f'pred {iteration}'].values, self.data[f'adj {iteration}'].values
        sns.scatterplot(x=x, y=pred, color='red', alpha=0.6, label='pred').set_xticks([])
        sns.scatterplot(x=x, y=adj, color='blue', alpha=0.6, label='adj')
        for i in x:
            plt.plot([i, i], [pred[i], adj[i]], alpha=0.6, c='black')
        return f'{iteration}) avg. distance = {np.abs(pred - adj).mean():.4f}'


class BoundsAnalysis(AnalysisCallback):
    def __init__(self, scalers, num_columns=1, **kwargs):
        super(BoundsAnalysis, self).__init__(scalers=scalers, num_columns=num_columns, **kwargs)

    def on_process_start(self, macs, x, y, val_data):
        super(BoundsAnalysis, self).on_process_start(macs, x, y, val_data)
        hi, li = macs.master.higher_indices, macs.master.lower_indices
        self.data['lower'] = self.data.index.map(lambda i: li[hi == i])
        self.data['higher'] = self.data.index.map(lambda i: hi[li == i])

    def on_pretraining_end(self, macs, x, y, val_data):
        pass

    def on_training_end(self, macs, x, y, val_data, iteration):
        self._insert_bounds(self.y_scaler.invert(macs.predict(x)), 'pred', iteration)

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        self._insert_bounds(self.y_scaler.invert(adjusted_y), 'adj', iteration)

    def _insert_bounds(self, v, label, iteration):
        self.data[f'{label} {iteration}'] = v
        self.data[f'{label} lb {iteration}'] = self.data['lower'].map(lambda i: v[i].max() if len(i) > 0 else None)
        self.data[f'{label} ub {iteration}'] = self.data['higher'].map(lambda i: v[i].min() if len(i) > 0 else None)

    def plot_function(self, iteration):
        x = np.arange(len(self.data))
        avg_bound = {}
        for label, color in dict(adj='blue', pred='red').items():
            val = self.data[f'{label} {iteration}']
            lb = self.data[f'{label} lb {iteration}'].fillna(val.min())
            ub = self.data[f'{label} ub {iteration}'].fillna(val.max())
            sns.scatterplot(x=x, y=lb, marker='^', color=color, alpha=0.4)
            sns.scatterplot(x=x, y=ub, marker='v', color=color, alpha=0.4)
            sns.scatterplot(x=x, y=val, color=color, edgecolors='black', label=label).set_xticks([])
            avg_bound[label] = np.mean(ub - lb)
        return f'{iteration}) ' + ', '.join([f'{k} bound = {v:.2f}' for k, v in avg_bound.items()])


class SyntheticAdjustments(AnalysisCallback):
    ground_size = 0.3
    alpha = 0.4

    def on_process_start(self, macs, x, y, val_data):
        super(SyntheticAdjustments, self).on_process_start(macs, x, y, val_data)
        self.data['ground'] = synthetic_function(self.data['a'], self.data['b'])

    def on_training_end(self, macs, x, y, val_data, iteration):
        self.data[f'pred {iteration}'] = self.y_scaler.invert(macs.predict(x))
        self.data[f'pred err {iteration}'] = self.data[f'pred {iteration}'] - self.data['ground']

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        self.data[f'adj {iteration}'] = self.y_scaler.invert(adjusted_y)
        self.data[f'adj err {iteration}'] = self.data[f'adj {iteration}'] - self.data['ground']
        self.data[f'sw {iteration}'] = kwargs.get('sample_weight', SyntheticAdjustments.ground_size * np.ones_like(y))

    def plot_function(self, iteration):
        def synthetic_inverse(column):
            b = np.sin(np.pi * (self.data['b'] - 0.01)) ** 2 + 1
            return (self.data[column] - b) * b

        a, sw, pred = self.data['a'], self.data[f'sw {iteration}'], synthetic_inverse(f'pred {iteration}')
        s, p = ['aug' if b else 'label' for b in np.isnan(self.data['label'])], dict(aug='blue', label='black')
        gs, al = SyntheticAdjustments.ground_size, SyntheticAdjustments.alpha
        sns.lineplot(x=self.data['a'], y=synthetic_inverse('ground'), label='ground', color='green')
        sns.scatterplot(x=a, y=pred, label='predictions', color='red', alpha=al, s=gs * 50)
        if iteration == PRETRAINING:
            adj = synthetic_inverse('label')
        else:
            adj = synthetic_inverse(f'adj {iteration}')
        sw[np.array(s) == 'label'] = gs
        sns.scatterplot(x=a, y=adj, hue=s, style=s, size=sw, size_norm=(0, 1), sizes=(0, 100), palette=p, alpha=al)
        plt.legend(['predictions', 'labels' if iteration == PRETRAINING else 'adjusted'])


class SyntheticResponse(AnalysisCallback):
    def __init__(self, scalers, res=50, **kwargs):
        super(SyntheticResponse, self).__init__(scalers=scalers, **kwargs)
        a, b = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
        self.grid = pd.DataFrame.from_dict({'a': a.flatten(), 'b': b.flatten()})
        self.fader = ColorFader('red', 'blue', bounds=(-1, 1))

    def on_training_end(self, macs, x, y, val_data, iteration):
        input_grid = self.x_scaler.transform(self.grid[['a', 'b']])
        self.grid[f'pred {iteration}'] = self.y_scaler.invert(macs.predict(input_grid))

    def plot_function(self, iteration):
        for idx, group in self.grid.groupby('b'):
            label = f'b = {idx:.0f}' if idx in [-1, 1] else None
            sns.lineplot(data=group, x='a', y=f'pred {iteration}', color=self.fader(idx), alpha=0.4, label=label)


class CarsAdjustments(AnalysisCallback):
    labels_size = 0.4
    alpha = 0.4

    def __init__(self, scalers, plot_kind='scatter', **kwargs):
        super(CarsAdjustments, self).__init__(scalers=scalers, **kwargs)
        assert plot_kind in ['line', 'scatter'], "plot_kind should be either 'line' or 'scatter'"
        self.plot_kind = plot_kind

    def on_training_end(self, macs, x, y, val_data, iteration):
        self.data[f'pred {iteration}'] = self.y_scaler.invert(macs.predict(x))

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        self.data[f'adj {iteration}'] = self.y_scaler.invert(adjusted_y)
        self.data[f'sw {iteration}'] = kwargs.get('sample_weight', CarsAdjustments.labels_size * np.ones_like(y))

    def plot_function(self, iteration):
        x, y = self.data['price'].values, self.data['sales'].values
        s, p = ['aug' if b else 'labels' for b in np.isnan(y)], dict(aug='blue', labels='black')
        gs, al = CarsAdjustments.labels_size, CarsAdjustments.alpha
        pred, adj = self.data[f'pred {iteration}'], self.data[f'adj {iteration}'],
        if iteration == PRETRAINING:
            sns.scatterplot(x=x, y=y, hue=s, style=s, size=gs, size_norm=(0, 1), sizes=(0, 100), palette=p, alpha=al)
        elif self.plot_kind == 'line':
            sns.lineplot(x=x, y=adj, color='blue')
            for i in range(self.data.shape[0]):
                plt.plot([x[i], x[i]], [y[i], adj[i]], c='black', alpha=al)
        elif self.plot_kind == 'scatter':
            sw = self.data[f'sw {iteration}']
            sw[np.array(s) == 'labels'] = gs
            sns.scatterplot(x=x, y=adj, hue=s, style=s, size=sw, size_norm=(0, 1), sizes=(0, 100), palette=p, alpha=al)
        sns.lineplot(x=x, y=pred, color='red')
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

    def on_training_end(self, macs, x, y, val_data, iteration):
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
