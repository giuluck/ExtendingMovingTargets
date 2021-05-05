import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from moving_targets.callbacks import Callback

PRETRAINING = 'PT'


class AnalysisCallback(Callback):
    def __init__(self, num_columns=5, sorting_attribute=None, file_signature=None, do_plot=True, **kwargs):
        super(AnalysisCallback, self).__init__()
        self.num_columns = num_columns
        self.sorting_attribute = sorting_attribute
        self.file_signature = file_signature
        self.do_plot = do_plot
        self.plot_kwargs = {'figsize': (20, 10), 'tight_layout': True}
        self.plot_kwargs.update(kwargs)
        self.data = None
        self.iterations = []

    def on_process_start(self, macs, x, y, val_data, **kwargs):
        m = pd.Series(['aug' if m else 'label' for m in macs.master.augmented_mask], name='mask')
        self.data = pd.concat((x, y, m), axis=1)

    def on_pretraining_start(self, macs, x, y, val_data, **kwargs):
        kwargs['iteration'] = PRETRAINING
        self.on_iteration_start(macs, x, y, val_data, **kwargs)
        self.on_adjustment_start(macs, x, y, val_data, **kwargs)
        self.on_adjustment_end(macs, x, y, np.ones_like(y) * np.nan, val_data, **kwargs)
        self.on_training_start(macs, x, y, val_data, **kwargs)

    def on_pretraining_end(self, macs, x, y, val_data, **kwargs):
        kwargs['iteration'] = PRETRAINING
        self.on_training_end(macs, x, y, val_data, **kwargs)
        self.on_iteration_end(macs, x, y, val_data, **kwargs)

    def on_iteration_start(self, macs, x, y, val_data, iteration, **kwargs):
        self.iterations.append(iteration)
        self.data[f'y {iteration}'] = y

    def on_process_end(self, macs, val_data, **kwargs):
        # sort values
        if self.sorting_attribute is not None:
            self.data = self.data.sort_values(self.sorting_attribute)
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


class DistanceAnalysis(AnalysisCallback):
    def __init__(self, ground_only=True, num_columns=1, **kwargs):
        super(DistanceAnalysis, self).__init__(num_columns=num_columns, **kwargs)
        self.ground_only = ground_only
        self.y = None

    def on_pretraining_start(self, macs, x, y, val_data, **kwargs):
        self.y = y.name
        super(DistanceAnalysis, self).on_pretraining_start(macs, x, y, val_data, **kwargs)

    def on_training_end(self, macs, x, y, val_data, iteration, **kwargs):
        self.data[f'pred {iteration}'] = macs.predict(x)

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        self.data[f'adj {iteration}'] = adjusted_y

    def on_process_end(self, macs, val_data, **kwargs):
        if self.ground_only:
            self.data = self.data[self.data['mask'] == 'label']
        super(DistanceAnalysis, self).on_process_end(macs, val_data)

    def plot_function(self, iteration):
        x = np.arange(len(self.data))
        y, p, j = self.data[self.y].values, self.data[f'pred {iteration}'].values, self.data[f'adj {iteration}'].values
        sns.scatterplot(x=x, y=y, color='black', alpha=0.6).set_xticks([])
        sns.scatterplot(x=x, y=p, color='red', alpha=0.6)
        s, m = self.data['mask'], dict(aug='o', label='X')
        sns.scatterplot(x=x, y=j, style=s, markers=m, color='blue', alpha=0.8, s=50)
        plt.legend(['labels', 'predictions', 'adjusted'])
        for i in x:
            plt.plot([i, i], [p[i], j[i]], c='red')
            plt.plot([i, i], [y[i], j[i]], c='black')
        avg_pred_distance = np.abs(p - j).mean()
        avg_label_distance = np.abs(y[s == 'label'] - j[s == 'label']).mean()
        return f'{iteration}) pred. distance = {avg_pred_distance:.4f}, label distance = {avg_label_distance:.4f}'


class BoundsAnalysis(AnalysisCallback):
    def __init__(self, num_columns=1, **kwargs):
        super(BoundsAnalysis, self).__init__(num_columns=num_columns, **kwargs)

    def on_process_start(self, macs, x, y, val_data, **kwargs):
        super(BoundsAnalysis, self).on_process_start(macs, x, y, val_data, **kwargs)
        hi, li = macs.master.higher_indices, macs.master.lower_indices
        self.data['lower'] = self.data.index.map(lambda i: li[hi == i])
        self.data['higher'] = self.data.index.map(lambda i: hi[li == i])

    def on_pretraining_end(self, macs, x, y, val_data, **kwargs):
        pass

    def on_training_end(self, macs, x, y, val_data, iteration, **kwargs):
        self._insert_bounds(macs.predict(x), 'pred', iteration)

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        self._insert_bounds(adjusted_y, 'adj', iteration)

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
