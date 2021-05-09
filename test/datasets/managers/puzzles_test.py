import numpy as np
import pandas as pd
import seaborn as sns

from moving_targets.metrics import MAE, MSE, R2
from src.datasets import PuzzlesManager
from src.models import MTRegressionMaster
from src.util.plot import ColorFader
from test.datasets.managers.test_manager import TestManager, AnalysisCallback


class PuzzlesTest(TestManager):
    def __init__(self, filepath='../../res/puzzles.csv', extrapolation=False, warm_start=False, **kwargs):
        super(PuzzlesTest, self).__init__(
            dataset=PuzzlesManager(filepath=filepath),
            master_type=MTRegressionMaster,
            metrics=[MAE(), MSE(), R2()],
            data_args=dict(extrapolation=extrapolation),
            augmented_args=dict(num_random=465, num_augmented=[3, 4, 8]),
            monotonicities_args=dict(kind='group'),
            learner_args=dict(output_act=None, h_units=[16] * 4, optimizer='adam', loss='mse', warm_start=warm_start),
            **kwargs
        )


class PuzzlesResponse(AnalysisCallback):
    features = ['word_count', 'star_rating', 'num_reviews']

    def __init__(self, feature, res=5, **kwargs):
        super(PuzzlesResponse, self).__init__(**kwargs)
        assert feature in self.features, f"feature should be in {self.features}"
        grid = np.meshgrid(np.linspace(0, 230, res), np.linspace(0, 5, res), np.linspace(0, 70, res))
        self.grid = pd.DataFrame.from_dict({k: v.flatten() for k, v in zip(self.features, grid)})
        self.feature = feature

    def on_training_end(self, macs, x, y, val_data, iteration, **kwargs):
        self.grid[f'pred {iteration}'] = macs.predict(self.grid[self.features])

    def plot_function(self, iteration):
        fi, fj = [f for f in self.features if f != self.feature]
        li, ui = self.grid[fi].min(), self.grid[fi].max()
        lj, uj = self.grid[fj].min(), self.grid[fj].max()
        fader = ColorFader('black', 'magenta', 'cyan', 'yellow', bounds=(li, lj, ui, uj))
        for (i, j), group in self.grid.groupby([fi, fj]):
            label = f'{fi}: {i:.0f}, {fj}: {j:.0f}' if (i in [li, ui] and j in [lj, uj]) else None
            sns.lineplot(data=group, x=self.feature, y=f'pred {iteration}', color=fader(i, j), alpha=0.6, label=label)
        return f'{iteration}) {self.feature.replace("_", " ").upper()}'
