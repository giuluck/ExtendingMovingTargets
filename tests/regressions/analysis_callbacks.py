import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from moving_targets.callbacks import Callback
from src.regressions import synthetic_function
from src.util.plot import ColorFader


class AnalysisCallback(Callback):
    def __init__(self, scalers, n_columns=5, plot=True, plot_fns=None, folder='../../temp/', **kwargs):
        super(AnalysisCallback, self).__init__()
        self.x_scaler = scalers[0]
        self.y_scaler = scalers[1]
        self.n_columns = n_columns
        self.plot = plot
        self.plot_fns = [fn for fn in dir(self) if '_plot' in fn] if plot_fns is None else plot_fns
        self.plot_fns = [getattr(self, fn) for fn in self.plot_fns]
        self.plot_kwargs = {'figsize': (20, 10), 'tight_layout': True}
        self.plot_kwargs.update(kwargs)
        self.folder = folder
        self.data = None
        self.iterations = []

    def on_process_start(self, macs, x, y, val_data):
        x = self.x_scaler.invert(x)
        y = self.y_scaler.invert(y)
        self.data = pd.concat((x, y), axis=1)

    def on_pretraining_end(self, macs, x, y, val_data):
        self.on_training_end(macs, x, y, val_data, 'pretraining')
        self.on_adjustment_end(macs, x, y, np.ones_like(y) * np.nan, val_data, 'pretraining')
        self.on_iteration_end(macs, x, y, val_data, 'pretraining')

    def on_training_end(self, macs, x, y, val_data, iteration):
        pred = macs.predict(x)
        self.data[f'pred {iteration}'] = self.y_scaler.invert(pred)

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration):
        adjusted_y = self.y_scaler.invert(adjusted_y)
        self.data[f'adj {iteration}'] = adjusted_y

    def on_iteration_end(self, macs, x, y, val_data, iteration):
        self.iterations.append(iteration)

    def on_process_end(self, macs, x, y, val_data):
        self.data = self.ordering_function()
        self.data.to_csv(self.folder + 'analysis.csv', index=False)
        with open(self.folder + 'analysis.txt', 'w') as f:
            f.write(str(self.data))
        if self.plot:
            for plot_fn in self.plot_fns:
                plt.figure(**self.plot_kwargs)
                n_rows = int(np.ceil(len(self.iterations) / self.n_columns))
                ax = None
                for idx, it in enumerate(self.iterations):
                    ax = plt.subplot(n_rows, self.n_columns, idx + 1, sharex=ax, sharey=ax)
                    title = plot_fn(it)
                    ax.set(xlabel='', ylabel='')
                    ax.set_title(f'{it})' if title is None else title)
                plt.show()

    def ordering_function(self):
        return self.data


class SyntheticAnalysis(AnalysisCallback):
    def __init__(self, scalers, n_columns=5, plot=True, plot_fns=None, folder='../../temp/', res=50, **kwargs):
        super(SyntheticAnalysis, self).__init__(scalers=scalers, n_columns=n_columns, plot=plot, plot_fns=plot_fns,
                                                folder=folder, **kwargs)
        a, b = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
        self.grid = pd.DataFrame.from_dict({'a': a.flatten(), 'b': b.flatten()})
        self.fader = ColorFader('red', 'blue', bounds=(-1, 1))

    def on_process_start(self, macs, x, y, val_data):
        super(SyntheticAnalysis, self).on_process_start(macs, x, y, val_data)
        self.data['ground'] = synthetic_function(self.data['a'], self.data['b'])

    def on_training_end(self, macs, x, y, val_data, iteration):
        super(SyntheticAnalysis, self).on_training_end(macs, x, y, val_data, iteration)
        input_grid = self.x_scaler.transform(self.grid[['a', 'b']])
        self.grid[f'pred {iteration}'] = self.y_scaler.invert(macs.predict(input_grid))
        self.data[f'pred err {iteration}'] = self.data[f'pred {iteration}'] - self.data['ground']

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration):
        super(SyntheticAnalysis, self).on_adjustment_end(macs, x, y, adjusted_y, val_data, iteration)
        self.data[f'adj err {iteration}'] = self.data[f'adj {iteration}'] - self.data['ground']

    def _analysis_plot(self, iteration):
        def synthetic_inverse(column):
            b = np.sin(np.pi * (self.data['b'] - 0.01)) ** 2 + 1
            return (self.data[column] - b) * b

        ground = synthetic_inverse('ground')
        pred = synthetic_inverse(f'pred {iteration}')
        sns.lineplot(x=self.data['a'], y=ground, label='ground', color='black')
        if iteration == 'pretraining':
            label = synthetic_inverse('label')
            sns.scatterplot(x=self.data['a'], y=label, label='labels', color='blue', alpha=0.3, s=20)
        else:
            adj = synthetic_inverse(f'adj {iteration}')
            sns.scatterplot(x=self.data['a'], y=adj, label='adjusted', color='blue', alpha=0.3, s=20)
        sns.scatterplot(x=self.data['a'], y=pred, label='predictions', color='red', alpha=0.3, s=20)

    def _response_plot(self, iteration):
        for idx, group in self.grid.groupby('b'):
            label = f'b = {idx:.0f}' if idx in [-1, 1] else None
            sns.lineplot(data=group, x='a', y=f'pred {iteration}', color=self.fader(idx), alpha=0.4, label=label)


class CarsAnalysis(AnalysisCallback):
    def __init__(self, scalers, n_columns=5, adj_plot='scatter', plot=True, plot_fns=None, folder='../../temp/',
                 **kwargs):
        super(CarsAnalysis, self).__init__(scalers=scalers, n_columns=n_columns, plot=plot, plot_fns=plot_fns,
                                           folder=folder, **kwargs)
        assert adj_plot in ['line', 'scatter'], "adj_plot should be either 'line' or 'scatter'"
        self.adj_plot = adj_plot

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration):
        super(CarsAnalysis, self).on_adjustment_end(macs, x, y, adjusted_y, val_data, iteration)
        self.data[f'eps {iteration}'] = self.data[f'adj {iteration}'] - self.data['sales']
        self.data[f'eps {iteration}'] = self.data[f'eps {iteration}'].fillna(0.)

    def ordering_function(self):
        return self.data.sort_values('price', ignore_index=True)

    def _analysis_plot(self, iteration):
        x, y = self.data['price'].values, self.data['sales'].values
        p, adj = self.data[f'pred {iteration}'].values, self.data[f'adj {iteration}'].values
        if iteration == 'pretraining':
            sns.scatterplot(x=x, y=y, label='ground', color='blue', alpha=0.4, s=20)
        elif self.adj_plot == 'line':
            sns.lineplot(x=x, y=adj, label='adjusted', color='blue')
            for i in range(self.data.shape[0]):
                plt.plot([x[i], x[i]], [y[i], adj[i]], c='black', alpha=0.4)
        elif self.adj_plot == 'scatter':
            sns.scatterplot(x=x, y=adj, label='adjusted', color='blue', alpha=0.4, s=20)
        sns.lineplot(x=x, y=p, label='predictions', color='red')
        return f'{iteration}) adj. mae = {np.abs(self.data[f"eps {iteration}"]).mean():.4f}'
