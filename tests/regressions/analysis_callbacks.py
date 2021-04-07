import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from moving_targets.callbacks import Callback


class CarsCallback(Callback):
    def __init__(self, n_columns=4, plot=True, **kwargs):
        super(CarsCallback, self).__init__()
        self.n_columns = n_columns
        self.plot = plot
        self.plot_kwargs = {'tight_layout': True}
        self.plot_kwargs.update(kwargs)
        self.data = pd.DataFrame()
        self.iterations = []

    def on_process_start(self, macs, x, y, val_data):
        self.data['price'] = x.flatten()
        self.data['sales'] = y

    def on_pretraining_end(self, macs, x, y, val_data):
        self.on_training_end(macs, x, y, val_data, 'pretraining')
        self.on_adjustment_end(macs, x, y, np.ones_like(y) * np.nan, val_data, 'pretraining')
        self.on_iteration_end(macs, x, y, val_data, 'pretraining')

    def on_training_end(self, macs, x, y, val_data, iteration):
        self.data[f'pred {iteration}'] = macs.predict(x)

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration):
        self.data[f'adj {iteration}'] = adjusted_y
        self.data[f'eps {iteration}'] = adjusted_y - y

    def on_iteration_end(self, macs, x, y, val_data, iteration):
        self.iterations.append(iteration)

    def on_process_end(self, macs, x, y, val_data):
        self.data = self.data.sort_values('price', ignore_index=True)
        if self.plot:
            # handle plot configuration
            n_rows = int(np.ceil(len(self.iterations) / self.n_columns))
            plt.figure(**self.plot_kwargs)
            # range over each iteration
            x, y = self.data['price'].values, self.data['sales'].values
            for idx, it in enumerate(self.iterations):
                title = f'{it}) adj. mae = {np.abs(self.data[f"eps {it}"]).mean():.4f}'
                p, adj = self.data[f'pred {it}'].values, self.data[f'adj {it}'].values
                ax = plt.subplot(n_rows, self.n_columns, idx + 1)
                if np.isnan(adj).all():
                    sns.scatterplot(x=x, y=y, color='black', alpha=0.4, s=25)
                else:
                    for i in range(self.data.shape[0]):
                        plt.plot([x[i], x[i]], [y[i], adj[i]], c='black', alpha=0.4)
                    sns.lineplot(x=x, y=adj, label='adjusted', color='blue', ax=ax)
                sns.lineplot(x=x, y=p, label='predictions', color='red', ax=ax).set(title=title, xlabel='', ylabel='')
            plt.show()
