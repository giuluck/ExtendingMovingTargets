import numpy as np
import pandas as pd
import seaborn as sns

from moving_targets.metrics import Accuracy
from src.datasets import DefaultManager
from test.datasets.managers.test_manager import ClassificationTest, AnalysisCallback


class DefaultTest(ClassificationTest):
    def __init__(self, kind='probabilities', filepath='../../res/default.csv', test_size=0.8, **kwargs):
        super(DefaultTest, self).__init__(
            kind=kind,
            h_units=[128, 128],
            evaluation_metric=Accuracy(name='metric'),
            dataset=DefaultManager(filepath=filepath, test_size=test_size),
            augmented_args=dict(num_augmented=15),
            monotonicities_args=dict(kind='group'),
            **kwargs
        )


class DefaultAdjustments(AnalysisCallback):
    max_size = 30
    alpha = 0.8

    def __init__(self, **kwargs):
        super(DefaultAdjustments, self).__init__(**kwargs)
        married, payment = np.meshgrid([0, 1], np.arange(-2, 9))
        self.grid = pd.DataFrame.from_dict({'married': married.flatten(), 'payment': payment.flatten()})

    def on_training_end(self, macs, x, y, val_data, iteration, **kwargs):
        self.data[f'pred {iteration}'] = macs.predict(x)
        self.grid[f'pred {iteration}'] = macs.predict(self.grid[['married', 'payment']])

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        self.data[f'adj {iteration}'] = adjusted_y
        self.data[f'sw {iteration}'] = kwargs.get('sample_weight', np.where(self.data['mask'] == 'label', 1, 0))

    def plot_function(self, iteration):
        y, adj = self.data['default'], self.data[f'adj {iteration}']
        label = 'default' if iteration == AnalysisCallback.PRETRAINING else f'adj {iteration}'
        data = self.data.astype({'married': int}).rename(columns={label: 'y', f'sw {iteration}': 'sw'})
        data = data.groupby(['married', 'payment', 'y'])['sw'].sum().reset_index()
        # dodge based on married value due to visualization reasons
        data['payment'] = [d + (0.1 if m == 0 else -0.1) for d, m in zip(data['payment'], data['married'])]
        # plot mass probabilities and responses
        sns.scatterplot(data=data, x='payment', y='y', hue='married', size='sw', size_norm=(0, 100), legend=False,
                        sizes=(0, DefaultAdjustments.max_size), alpha=DefaultAdjustments.alpha)
        sns.lineplot(data=self.grid, x='payment', y=f'pred {iteration}', hue='married')
        return f'{iteration}) avg. flips = {np.abs(adj[~np.isnan(y)] - y[~np.isnan(y)]).mean():.4f}'
