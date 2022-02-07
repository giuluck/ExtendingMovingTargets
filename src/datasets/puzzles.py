import importlib.resources
from typing import Dict, Callable, Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.datasets.manager import Manager, AnalysisCallback
from src.util.analysis import ColorFader


class Puzzles(Manager):
    """Data Manager for the Puzzles Dataset."""

    __name__: str = 'cars'

    callbacks: Dict[str, Callable] = {
        **Manager.callbacks,
        'response_wc': lambda fs: PuzzlesResponse(feature='word_count', file_signature=fs),
        'response_sr': lambda fs: PuzzlesResponse(feature='star_rating', file_signature=fs),
        'response_nr': lambda fs: PuzzlesResponse(feature='num_reviews', file_signature=fs)
    }

    @classmethod
    def load(cls) -> Dict[str, pd.DataFrame]:
        with importlib.resources.path('res', 'puzzles.csv') as filepath:
            df = pd.read_csv(filepath)
        mask = df['split']
        df = df[['word_count', 'star_rating', 'num_reviews', 'label']]
        return {split: df[mask == split] for split in ['train', 'test']}

    @classmethod
    def grid(cls, plot: bool = True) -> pd.DataFrame:
        res = 5 if plot else 20
        wc, sr, nr = np.meshgrid(np.linspace(0, 230, res), np.linspace(0, 5, res), np.linspace(0, 70, res))
        return pd.DataFrame({'word_count': wc.flatten(), 'star_rating': sr.flatten(), 'num_reviews': nr.flatten()})

    def __init__(self):
        super(Puzzles, self).__init__(label='label',
                                      classification=False,
                                      directions={'word_count': -1, 'star_rating': 1, 'num_reviews': 1})

    def _plot(self, model):
        fig, axes = plt.subplots(1, 3, sharey='all', figsize=(16, 9), tight_layout=True)
        grid = self.grid(plot=True)
        grid['pred'] = model.predict(grid).flatten()
        for ax, feat in zip(axes, ['word_count', 'star_rating', 'num_reviews']):
            # plot predictions for each group of other features
            fi, fj = [f for f in grid.columns if f not in [feat, 'pred']]
            li, ui = grid[fi].min(), grid[fi].max()
            lj, uj = grid[fj].min(), grid[fj].max()
            fader = ColorFader('black', 'magenta', 'cyan', 'yellow', bounds=[li, lj, ui, uj])
            for (i, j), group in grid.groupby([fi, fj]):
                label = f'{fi}: {i:.0f}, {fj}: {j:.0f}' if (i in [li, ui] and j in [lj, uj]) else None
                sns.lineplot(data=group, x=feat, y='pred', color=fader(i, j), alpha=0.6, label=label, ax=ax)
        fig.suptitle('Estimated Functions')


class PuzzlesResponse(AnalysisCallback):
    """Investigates marginal feature responses during iterations in puzzles dataset."""

    features = ['word_count', 'star_rating', 'num_reviews']

    def __init__(self,
                 feature: str,
                 sorting_attribute: Optional[str] = None,
                 file_signature: Optional[str] = None,
                 num_columns: Union[int, str] = 'auto'):
        super(PuzzlesResponse, self).__init__(sorting_attribute=sorting_attribute,
                                              file_signature=file_signature,
                                              num_columns=num_columns)
        assert feature in self.features, f"'{feature}' is not a valid feature"
        self.grid: pd.DataFrame = Puzzles.grid(plot=True)
        self.feature: str = feature

    def on_training_end(self, macs, x, y, val_data):
        self.grid[f'pred {macs.iteration}'] = macs.predict(self.grid[self.features])

    def _plot_function(self, iteration: int) -> Optional[str]:
        fi, fj = [f for f in self.features if f != self.feature]
        li, ui = self.grid[fi].min(), self.grid[fi].max()
        lj, uj = self.grid[fj].min(), self.grid[fj].max()
        fader = ColorFader('black', 'magenta', 'cyan', 'yellow', bounds=[li, lj, ui, uj])
        for (i, j), group in self.grid.groupby([fi, fj]):
            label = f'{fi}: {i:.0f}, {fj}: {j:.0f}' if (i in [li, ui] and j in [lj, uj]) else None
            sns.lineplot(data=group, x=self.feature, y=f'pred {iteration}', color=fader(i, j), alpha=0.6, label=label)
        return f'{iteration}) {self.feature.replace("_", " ").upper()}'
