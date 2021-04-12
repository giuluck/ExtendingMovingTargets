import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from moving_targets.callbacks import Callback

PRETRAINING = 'PT'


class AnalysisCallback(Callback):
    def __init__(self, scalers, num_columns=5, sorting_attributes=None, file_signature='../../temp/analysis',
                 do_plot=True, **kwargs):
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


class BoundsCallback(AnalysisCallback):
    def __init__(self, scalers, num_columns=1, **kwargs):
        super(BoundsCallback, self).__init__(scalers=scalers, num_columns=num_columns, **kwargs)

    def on_process_start(self, macs, x, y, val_data):
        super(BoundsCallback, self).on_process_start(macs, x, y, val_data)
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


class GroundCallback(AnalysisCallback):
    def __init__(self, scalers, num_columns=1, **kwargs):
        super(GroundCallback, self).__init__(scalers=scalers, num_columns=num_columns, **kwargs)

    def on_pretraining_end(self, macs, x, y, val_data):
        pass

    def on_training_end(self, macs, x, y, val_data, iteration):
        self.data[f'pred {iteration}'] = self.y_scaler.invert(macs.predict(x))

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        self.data[f'adj {iteration}'] = self.y_scaler.invert(adjusted_y)

    def on_process_end(self, macs, x, y, val_data):
        self.data = self.data[~np.isnan(y)]
        super(GroundCallback, self).on_process_end(macs, x, y, val_data)

    def plot_function(self, iteration):
        x = np.arange(len(self.data))
        pred, adj = self.data[f'pred {iteration}'].values, self.data[f'adj {iteration}'].values
        sns.scatterplot(x=x, y=pred, color='red', alpha=0.6, label='pred').set_xticks([])
        sns.scatterplot(x=x, y=adj, color='blue', alpha=0.6, label='adj')
        for i in x:
            plt.plot([i, i], [pred[i], adj[i]], alpha=0.6, c='black')
        return f'{iteration}) avg. distance = {np.abs(pred - adj).mean():.4f}'
