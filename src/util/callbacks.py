from typing import List, Any, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from moving_targets.callbacks import Callback

from src.datasets import Dataset
from src.util.analysis import set_pandas_options


class AnalysisCallback(Callback):
    """Template callback for results analysis and plotting.

    - 'sorting_attribute' is string representing the (optional) attribute on which to sort the output data.
    - 'file_signature' is the (optional) file signature without extension where to store the output data.
    - 'num_columns' is the (optional) number of columns in the final plot. If None, does not plot the results.
    """

    def __init__(self,
                 sorting_attribute: Optional[str] = None,
                 file_signature: Optional[str] = None,
                 num_columns: Optional[int] = None,
                 figsize: Tuple[int, int] = (16, 9),
                 tight_layout: bool = True,
                 **plt_kwargs):
        super(AnalysisCallback, self).__init__()
        self.sorting_attribute: Optional[str] = sorting_attribute
        self.file_signature: Optional[str] = file_signature
        self.num_columns: Optional[int] = num_columns
        self.plt_kwargs: Dict[str, Any] = {'figsize': figsize, 'tight_layout': tight_layout, **plt_kwargs}
        self.data: Optional[pd.DataFrame] = None
        self.iterations: List[Any] = []

    def on_process_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self.data = pd.concat((x.reset_index(drop=True), pd.Series(y, name='y')), axis=1)

    def on_pretraining_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self.on_iteration_start(macs, x, y, val_data)
        self.on_adjustment_start(macs, x, y, val_data)
        self.on_adjustment_end(macs, x, y, np.ones_like(y) * np.nan, val_data)
        self.on_training_start(macs, x, y, val_data)

    def on_pretraining_end(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self.on_training_end(macs, x, y, val_data)
        self.on_iteration_end(macs, x, y, val_data)

    def on_training_end(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self.data[f'pred {macs.iteration}'] = macs.predict(x)

    def on_adjustment_end(self, macs, x, y: np.ndarray, adjusted_y: np.ndarray, val_data: Optional[Dataset]):
        self.data[f'adj {macs.iteration}'] = adjusted_y

    def on_iteration_end(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self.iterations.append(macs.iteration)

    def on_process_end(self, macs, val_data: Optional[Dataset]):
        # sort values
        if self.sorting_attribute is not None:
            self.data = self.data.sort_values(self.sorting_attribute)
        # write on files
        if self.file_signature is not None:
            set_pandas_options()
            self.data.to_csv(f'{self.file_signature}.csv', index_label='index')
            with open(f'{self.file_signature}.txt', 'w') as f:
                f.write(str(self.data))
        # plots results
        plt.figure(**self.plt_kwargs)
        num_rows = int(np.ceil(len(self.iterations) / self.num_columns))
        ax = None
        for idx, it in enumerate(self.iterations):
            ax = plt.subplot(num_rows, self.num_columns, idx + 1, sharex=ax, sharey=ax)
            title = self._plot_function(it)
            ax.set(xlabel='', ylabel='')
            ax.set_title(f'{it})' if title is None else title)
        plt.show()

    def _plot_function(self, iteration: Any) -> Optional[str]:
        x = np.arange(len(self.data))
        y, p, j = self.data['y'].values, self.data[f'pred {iteration}'].values, self.data[f'adj {iteration}'].values
        sns.scatterplot(x=x, y=y, color='black', alpha=0.6).set_xticks([])
        sns.scatterplot(x=x, y=p, color='red', alpha=0.6)
        sns.scatterplot(x=x, y=j, color='blue', alpha=0.8, s=50)
        plt.legend(['labels', 'predictions', 'adjusted'])
        for i in x:
            plt.plot([i, i], [p[i], j[i]], c='red')
            plt.plot([i, i], [y[i], j[i]], c='black')
        avg_pred_distance = np.abs(p - j).mean()
        avg_label_distance = np.abs(y - j).mean()
        return f'{iteration}) pred. distance = {avg_pred_distance:.4f}, label distance = {avg_label_distance:.4f}'
