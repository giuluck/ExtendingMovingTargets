"""Law Test Manager & Callbacks."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Optional as Opt

from moving_targets.metrics import Accuracy
from moving_targets.util.typing import Matrix, Vector, Dataset, Iteration
from src.datasets import LawManager
from src.util.plot import ColorFader
from experimental.datasets.managers.test_manager import ClassificationTest, AnalysisCallback


# noinspection PyMissingOrEmptyDocstring
class LawTest(ClassificationTest):
    def __init__(self,
                 filepath: str = '../../res/law.csv',
                 dataset_args: Opt[dict] = None,
                 lrn_h_units: tuple = (128, 128),
                 **kwargs):
        super(LawTest, self).__init__(
            dataset=LawManager(filepath=filepath, **{} if dataset_args is None else dataset_args),
            lrn_h_units=lrn_h_units,
            mst_evaluation_metric=Accuracy(name='metric'),
            **kwargs
        )


# noinspection PyMissingOrEmptyDocstring
class LawAdjustments(AnalysisCallback):
    max_size = 40

    def __init__(self, res: int = 100, data_points: bool = True, **kwargs):
        super(LawAdjustments, self).__init__(**kwargs)
        lsat, ugpa = np.meshgrid(np.linspace(0, 50, res), np.linspace(0, 4, res))
        self.grid: pd.DataFrame = pd.DataFrame.from_dict({'lsat': lsat.flatten(), 'ugpa': ugpa.flatten()})
        self.res: int = res
        self.data_points: bool = data_points

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        grid = self.grid[['lsat', 'ugpa']]
        data = self.data[['lsat', 'ugpa']]
        self.grid[f'pred {iteration}'] = macs.predict(grid)
        self.data[f'pred {iteration}'] = macs.predict(data)

    def on_adjustment_end(self, macs, x: Matrix, y: Vector, adjusted_y: Vector, val_data: Opt[Dataset],
                          iteration: Iteration, **kwargs):
        self.data[f'adj {iteration}'] = adjusted_y
        self.data[f'sw {iteration}'] = kwargs.get('sample_weight', np.where(self.data['mask'] == 'label', 1, 0))

    def plot_function(self, iteration: Iteration) -> Opt[str]:
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
        return


# noinspection PyMissingOrEmptyDocstring
class LawResponse(AnalysisCallback):
    def __init__(self, feature: str, res: int = 100, **kwargs):
        super(LawResponse, self).__init__(**kwargs)
        assert feature in ['lsat', 'ugpa'], f"'{feature}' is not a valid feature"
        lsat, ugpa = np.meshgrid(np.linspace(0, 50, res), np.linspace(0, 4, res))
        self.grid: pd.DataFrame = pd.DataFrame.from_dict({'lsat': lsat.flatten(), 'ugpa': ugpa.flatten()})
        self.fader: ColorFader = ColorFader('red', 'blue', bounds=[0, 4] if feature == 'lsat' else [0, 50])
        self.features: Tuple[str, str] = ('lsat', 'ugpa') if feature == 'lsat' else ('ugpa', 'lsat')

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        input_grid = self.grid[['lsat', 'ugpa']]
        self.grid[f'pred {iteration}'] = macs.predict(input_grid)

    def plot_function(self, iteration: Iteration) -> Opt[str]:
        feat, group_feat = self.features
        for group_val, group in self.grid.groupby([group_feat]):
            sns.lineplot(data=group, x=feat, y=f'pred {iteration}', color=self.fader(group_val), alpha=0.6)
        return f'{iteration}) {feat.upper()}'
