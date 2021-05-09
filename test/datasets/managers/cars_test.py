import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from moving_targets.metrics import MAE, MSE, R2
from src.datasets import CarsManager
from src.models import MTRegressionMaster
from test.datasets.managers.test_manager import TestManager, AnalysisCallback


class AbstractCarsTest(TestManager):
    def __init__(self, augmented_args, monotonicities_args, filepath='../../res/cars.csv', extrapolation=False,
                 warm_start=False, **kwargs):
        super(AbstractCarsTest, self).__init__(
            dataset=CarsManager(filepath=filepath),
            master_type=MTRegressionMaster,
            metrics=[MAE(), MSE(), R2()],
            data_args=dict(extrapolation=extrapolation),
            augmented_args=augmented_args,
            monotonicities_args=monotonicities_args,
            learner_args=dict(output_act=None, h_units=[16] * 4, optimizer='adam', loss='mse', warm_start=warm_start),
            **kwargs
        )


class CarsTest(AbstractCarsTest):
    def __init__(self, **kwargs):
        super(CarsTest, self).__init__(
            augmented_args=dict(num_augmented=15),
            monotonicities_args=dict(kind='group'),
            **kwargs
        )


class CarsUnivariateTest(AbstractCarsTest):
    def __init__(self, **kwargs):
        super(CarsUnivariateTest, self).__init__(
            augmented_args=dict(num_augmented=0),
            monotonicities_args=dict(kind='all', errors='ignore'),
            **kwargs
        )


class CarsAdjustments(AnalysisCallback):
    label_size = 0.4
    max_size = 100
    alpha = 0.4

    def __init__(self, plot_kind='scatter', **kwargs):
        super(CarsAdjustments, self).__init__(**kwargs)
        assert plot_kind in ['line', 'scatter'], "plot_kind should be either 'line' or 'scatter'"
        self.plot_kind = plot_kind

    def on_training_end(self, macs, x, y, val_data, iteration, **kwargs):
        self.data[f'pred {iteration}'] = macs.predict(x)

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        self.data[f'adj {iteration}'] = adjusted_y
        self.data[f'sw {iteration}'] = kwargs.get('sample_weight', CarsAdjustments.label_size * np.ones_like(y))

    def plot_function(self, iteration):
        x, y = self.data['price'].values, self.data['sales'].values
        s, m = np.array(self.data['mask']), dict(aug='o', label='X')
        sn, al = (0, CarsAdjustments.max_size), CarsAdjustments.alpha
        p, adj, sw = self.data[f'pred {iteration}'], self.data[f'adj {iteration}'], self.data[f'sw {iteration}'].values
        # rescale in case of uniform values
        if np.allclose(sw, 1.0):
            sw *= CarsAdjustments.label_size
        else:
            sw[s == 'label'] = CarsAdjustments.label_size
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
        return f'{iteration}) adj. mae = {np.abs((adj - y).fillna(0)).mean():.4f}'
