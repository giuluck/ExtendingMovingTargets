import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from moving_targets.callbacks import Callback


class CarsCallback(Callback):
    def __init__(self, n_columns=3, plot=True, **kwargs):
        super(CarsCallback, self).__init__()
        self.n_columns = n_columns
        self.plot = plot
        self.plot_kwargs = {'tight_layout': True}
        self.plot_kwargs.update(kwargs)
        self.data = pd.DataFrame()

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration):
        if iteration == 1:
            self.data['price'] = x.flatten()
            self.data['sales'] = y
        self.data[f'pred {iteration}'] = macs.predict(x)
        self.data[f'adj {iteration}'] = adjusted_y
        self.data[f'eps {iteration}'] = adjusted_y - y

    def on_process_end(self, macs, x, y, val_data):
        self.data = self.data.sort_values('price', ignore_index=True)
        if self.plot:
            # handle plot configuration
            n_iterations = (self.data.shape[1] - 2) // 3
            n_rows = int(np.ceil(n_iterations / self.n_columns))
            plt.figure(**self.plot_kwargs)
            # range over each iteration
            x, y = self.data['price'].values, self.data['sales'].values
            for it in range(n_iterations):
                pred, adj = self.data[f'pred {it}'].values, self.data[f'adj {it}'].values
                adj_mae = np.abs(self.data[f'eps {it}']).mean()
                ax = plt.subplot(n_rows, self.n_columns, it + 1)
                for i in range(self.data.shape[0]):
                    plt.plot([x[i], x[i]], [y[i], adj[i]], c='black', alpha=0.4)
                sns.lineplot(x=x, y=adj, label='adjusted', color='red', ax=ax).set(title=f'Adj. MAE: {adj_mae}')
                sns.lineplot(x=x, y=pred, label='predictions', color='blue', ax=ax).set(xlabel='', ylabel='')
