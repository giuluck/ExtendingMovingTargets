"""Puzzles Test Manager & Callbacks."""

import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional as Opt

from moving_targets.util.typing import Matrix, Vector, Dataset, Iteration
from src.datasets import PuzzlesManager
from src.util.plot import ColorFader
from src.util.typing import Augmented
from test.datasets.managers.test_manager import RegressionTest, AnalysisCallback


# noinspection PyMissingOrEmptyDocstring
class PuzzlesTest(RegressionTest):
    def __init__(self,
                 filepath: str = '../../res/puzzles.csv',
                 lrn_h_units: tuple = (16,) * 4,
                 aug_num_augmented: Augmented = (3, 4, 8),
                 aug_num_random: int = 465,
                 **kwargs):
        super(PuzzlesTest, self).__init__(
            dataset=PuzzlesManager(filepath=filepath),
            lrn_h_units=lrn_h_units,
            aug_num_augmented=aug_num_augmented,
            aug_num_random=aug_num_random,
            **kwargs
        )


# noinspection PyMissingOrEmptyDocstring
class PuzzlesResponse(AnalysisCallback):
    features = ['word_count', 'star_rating', 'num_reviews']

    def __init__(self, feature: str, res: int = 5, **kwargs):
        super(PuzzlesResponse, self).__init__(**kwargs)
        assert feature in self.features, f"feature should be in {self.features}"
        grid = np.meshgrid(np.linspace(0, 230, res), np.linspace(0, 5, res), np.linspace(0, 70, res))
        self.grid: pd.DataFrame = pd.DataFrame.from_dict({k: v.flatten() for k, v in zip(self.features, grid)})
        self.feature: str = feature

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        self.grid[f'pred {iteration}'] = macs.predict(self.grid[self.features])

    def plot_function(self, iteration: Iteration) -> Opt[str]:
        fi, fj = [f for f in self.features if f != self.feature]
        li, ui = self.grid[fi].min(), self.grid[fi].max()
        lj, uj = self.grid[fj].min(), self.grid[fj].max()
        fader = ColorFader('black', 'magenta', 'cyan', 'yellow', bounds=[li, lj, ui, uj])
        for (i, j), group in self.grid.groupby([fi, fj]):
            label = f'{fi}: {i:.0f}, {fj}: {j:.0f}' if (i in [li, ui] and j in [lj, uj]) else None
            sns.lineplot(data=group, x=self.feature, y=f'pred {iteration}', color=fader(i, j), alpha=0.6, label=label)
        return f'{iteration}) {self.feature.replace("_", " ").upper()}'
