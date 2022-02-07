import importlib.resources
from typing import Dict, Callable, Optional, Union, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.datasets.manager import Manager, AnalysisCallback
from src.util.preprocessing import split_dataset


class Cars(Manager):
    """Data Manager for the Cars Dataset."""

    __name__: str = 'cars'

    callbacks: Dict[str, Callable] = {
        **Manager.callbacks,
        'adjustmentsL': lambda fs: CarsAdjustments(scatter_plot=False),
        'adjustmentsS': lambda fs: CarsAdjustments(scatter_plot=True)
    }

    @classmethod
    def load(cls) -> Dict[str, pd.DataFrame]:
        with importlib.resources.path('res', 'cars.csv') as filepath:
            df = pd.read_csv(filepath)
        df = df[['price', 'sales']].dropna()
        return split_dataset(df, test_size=0.2, val_size=0.0)

    @classmethod
    def grid(cls, plot: bool = True) -> pd.DataFrame:
        return pd.DataFrame.from_dict({'price': np.linspace(0, 100, 700)})

    def __init__(self):
        super(Cars, self).__init__(label='sales', directions={'price': -1}, classification=False)

    def _plot(self, model):
        plt.figure(figsize=(16, 9), tight_layout=True)
        for split, df in [('train', self.train), ('test', self.test)]:
            sns.scatterplot(x=df['price'], y=df['sales'], alpha=0.6, sizes=0.6, label=split.capitalize())
        x = self.grid().values
        y = model.predict(x)
        sns.lineplot(x=x.flatten(), y=y.flatten(), color='black').set(ylabel='sales', title='Estimated Function')
        plt.xlim(np.min(x), np.max(x))
        plt.ylim(np.min(y), np.max(y))


class CarsAdjustments(AnalysisCallback):
    """Investigates Moving Targets adjustments during iterations in cars dataset.

    - 'scatter_plot' is a boolean value representing whether to plot data as a scatter plot or as a line plot.
    """

    def __init__(self,
                 scatter_plot: bool = False,
                 file_signature: Optional[str] = None,
                 num_columns: Union[int, str] = 'auto',
                 figsize: Tuple[int, int] = (16, 9),
                 tight_layout: bool = True,
                 **plt_kwargs):
        super(CarsAdjustments, self).__init__(sorting_attribute='price',
                                              file_signature=file_signature,
                                              num_columns=num_columns,
                                              figsize=figsize,
                                              tight_layout=tight_layout,
                                              **plt_kwargs)
        self.scatter_plot: bool = scatter_plot

    def on_process_start(self, macs, x, y, val_data):
        self.data = x.reset_index(drop=True)
        self.data['y'] = pd.Series(y, name='y')

    def on_training_end(self, macs, x, y, val_data):
        self.data[f'pred {macs.iteration}'] = macs.predict(x)

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data):
        self.data[f'adj {macs.iteration}'] = adjusted_y

    def _plot_function(self, iteration: int) -> Optional[str]:
        x, y, p = self.data['price'].values, self.data['y'].values, self.data[f'pred {iteration}'].values,
        # rescale original labels weights
        if iteration == 0:
            mse, label = np.nan, 'labels'
            sns.scatterplot(x=x, y=y, color='black')
        else:
            j = self.data[f'adj {iteration}'].values
            mse, label = np.square(j - y).mean(), 'adjusted'
            if self.scatter_plot:
                sns.scatterplot(x=x, y=j, color='blue', alpha=0.4)
            else:
                sns.lineplot(x=x, y=j, color='blue', alpha=0.6)
        sns.lineplot(x=x, y=p, color='red')
        plt.legend(['predictions', label])
        return f'{iteration}) adj. mse = {mse:.4f}'
