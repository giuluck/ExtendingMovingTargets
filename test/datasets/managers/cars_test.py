from typing import Any, Tuple, Dict, Optional

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.datasets import CarsManager
from test.datasets.managers.test_manager import AnalysisCallback, RegressionTest


class CarsTest(RegressionTest):
    def __init__(self, filepath: str = '../../res/cars.csv', **kwargs):
        super(CarsTest, self).__init__(
            dataset=CarsManager(filepath=filepath),
            augmented_args=dict(num_augmented=15),
            monotonicities_args=dict(kind='group'),
            **kwargs
        )


class CarsUnivariateTest(RegressionTest):
    def __init__(self, filepath: str = '../../res/cars.csv', **kwargs):
        super(CarsUnivariateTest, self).__init__(
            dataset=CarsManager(filepath=filepath),
            augmented_args=dict(num_augmented=0),
            monotonicities_args=dict(kind='all', errors='ignore'),
            **kwargs
        )


class CarsAdjustments(AnalysisCallback):
    max_size = 50
    alpha = 0.4

    def __init__(self, plot_kind: str = 'scatter', **kwargs):
        super(CarsAdjustments, self).__init__(**kwargs)
        assert plot_kind in ['line', 'scatter'], "plot_kind should be either 'line' or 'scatter'"
        self.plot_kind: str = plot_kind

    def on_training_end(self, macs, x, y, val_data: Dict[str, Tuple[Any, Any]], iteration: Any, **kwargs):
        self.data[f'pred {iteration}'] = macs.predict(x)

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data: Dict[str, Tuple[Any, Any]], iteration: Any, **kwargs):
        self.data[f'adj {iteration}'] = adjusted_y
        self.data[f'sw {iteration}'] = kwargs.get('sample_weight', np.where(self.data['mask'] == 'label', 1, 0))

    def plot_function(self, iteration: Any) -> Optional[str]:
        x, y = self.data['price'].values, self.data['sales'].values
        s, m = np.array(self.data['mask']), dict(aug='o', label='X')
        sn, al = (0, CarsAdjustments.max_size), CarsAdjustments.alpha
        p, adj, sw = self.data[f'pred {iteration}'], self.data[f'adj {iteration}'], self.data[f'sw {iteration}'].values
        # rescale original labels weights
        if iteration == AnalysisCallback.PRETRAINING:
            sns.scatterplot(x=x, y=y, style=s, markers=m, size=sw, size_norm=(0, 1), sizes=sn, color='black', alpha=al)
        elif self.plot_kind == 'line':
            sns.lineplot(x=x, y=adj, color='blue')
            for i in range(self.data.shape[0]):
                plt.plot([x[i], x[i]], [y[i], adj[i]], c='black', alpha=al)
        elif self.plot_kind == 'scatter':
            sns.scatterplot(x=x, y=adj, style=s, markers=m, size=sw, size_norm=(0, 1), sizes=sn, color='blue', alpha=al)
        sns.lineplot(x=x, y=p, color='red')
        plt.legend(['predictions', 'labels' if iteration == AnalysisCallback.PRETRAINING else 'adjusted'])
        return f'{iteration}) adj. mse = {np.square((adj - y).fillna(0)).mean():.4f}'
