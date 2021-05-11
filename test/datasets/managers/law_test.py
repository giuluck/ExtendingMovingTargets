import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from moving_targets.metrics import CrossEntropy, Accuracy
from src.datasets import LawManager
from src.models import MTClassificationMaster
from src.util.plot import ColorFader
from test.datasets.managers.test_manager import TestManager, AnalysisCallback


class LawTest(TestManager):
    def __init__(self, filepath='../../res/law.csv', warm_start=False, **kwargs):
        super(LawTest, self).__init__(
            dataset=LawManager(filepath=filepath),
            master_type=MTClassificationMaster,
            metrics=[CrossEntropy(), Accuracy()],
            data_args=dict(),
            augmented_args=dict(num_augmented=15),
            monotonicities_args=dict(kind='group'),
            learner_args=dict(output_act='sigmoid', h_units=[128, 128], optimizer='adam', loss='binary_crossentropy',
                              warm_start=warm_start),
            **kwargs
        )


class LawAdjustments(AnalysisCallback):
    max_size = 40

    def __init__(self, res=100, data_points=True, **kwargs):
        super(LawAdjustments, self).__init__(**kwargs)
        lsat, ugpa = np.meshgrid(np.linspace(0, 50, res), np.linspace(0, 4, res))
        self.grid = pd.DataFrame.from_dict({'lsat': lsat.flatten(), 'ugpa': ugpa.flatten()})
        self.res = res
        self.data_points = data_points

    def on_training_end(self, macs, x, y, val_data, iteration, **kwargs):
        grid = self.grid[['lsat', 'ugpa']]
        data = self.data[['lsat', 'ugpa']]
        self.grid[f'pred {iteration}'] = macs.predict(grid)
        self.data[f'pred {iteration}'] = macs.predict(data)

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        self.data[f'adj {iteration}'] = adjusted_y
        self.data[f'sw {iteration}'] = kwargs.get('sample_weight', np.where(self.data['mask'] == 'label', 1, 0))

    def plot_function(self, iteration):
        # plot 3D response
        lsat = self.grid['lsat'].values.reshape(self.res, self.res)
        ugpa = self.grid['ugpa'].values.reshape(self.res, self.res)
        pred = self.grid[f'pred {iteration}'].values.reshape(self.res, self.res)
        plt.pcolor(lsat, ugpa, pred, shading='auto', vmin=0, vmax=1)
        # plot data points
        if self.data_points:
            markers, sizes = AnalysisCallback.MARKERS, (0, LawAdjustments.max_size)
            sns.scatterplot(data=self.data, x='lsat', y='ugpa', size=f'sw {iteration}', size_norm=(0, 1),
                            sizes=sizes, color='black', style='mask', markers=markers, legend=False)


class LawResponse(AnalysisCallback):
    def __init__(self, feature, res=100, **kwargs):
        super(LawResponse, self).__init__(**kwargs)
        assert feature in ['lsat', 'ugpa'], "feature should be either 'lsat' or 'ugpa'"
        lsat, ugpa = np.meshgrid(np.linspace(0, 50, res), np.linspace(0, 4, res))
        self.grid = pd.DataFrame.from_dict({'lsat': lsat.flatten(), 'ugpa': ugpa.flatten()})
        self.fader = ColorFader('red', 'blue', bounds=(0, 4) if feature == 'lsat' else (0, 50))
        self.features = ('lsat', 'ugpa') if feature == 'lsat' else ('ugpa', 'lsat')

    def on_training_end(self, macs, x, y, val_data, iteration, **kwargs):
        input_grid = self.grid[['lsat', 'ugpa']]
        self.grid[f'pred {iteration}'] = macs.predict(input_grid)

    def plot_function(self, iteration):
        feat, group_feat = self.features
        for group_val, group in self.grid.groupby([group_feat]):
            sns.lineplot(data=group, x=feat, y=f'pred {iteration}', color=self.fader(group_val), alpha=0.6)
        return f'{iteration}) {feat.upper()}'
