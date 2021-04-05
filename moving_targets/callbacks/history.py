import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from moving_targets.callbacks.logger import Logger


class History(Logger):
    def __init__(self):
        super(History, self).__init__()
        self.history = []

    def on_pretraining_end(self, macs, x, y, val_data):
        self.on_iteration_end(macs, x, y, val_data, -1)

    def on_iteration_end(self, macs, x, y, val_data, iteration):
        self.history.append(pd.DataFrame([self.cache.values()], columns=self.cache.keys(), index=[iteration + 1]))
        self.cache = {}

    def on_process_end(self, macs, x, y, val_data):
        self.history = pd.concat(self.history)

    def plot(self, columns=None, n_columns=3, show=True, **kwargs):
        # handle columns and number of rows
        assert isinstance(self.history, pd.DataFrame), 'Process did not end correctly therefore no dataframe was built'
        columns = self.history.columns if columns is None else columns
        columns = [c for c in columns if np.issubdtype(self.history[c].dtype, np.number)]
        n_rows = int(np.ceil(len(columns) / n_columns))
        # handle matplotlib arguments
        plt_args = dict(tight_layout=True)
        plt_args.update(kwargs)
        plt.figure(**plt_args)
        # plot each column in a subplot
        for idx, col in enumerate(columns):
            plt.subplot(n_rows, n_columns, idx + 1)
            p = sns.lineplot(x=self.history.index, y=self.history[col])
            p.set(title=col, xlabel='', ylabel='')
            p.set_xticks(self.history.index)
        # show plots
        if show:
            plt.show()
