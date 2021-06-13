"""Puzzles Data Manager."""

from typing import Optional, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

from src.datasets.abstract_manager import AbstractManager
from src.util.plot import ColorFader
from src.util.preprocessing import split_dataset
from src.util.typing import Methods, Augmented, Rng, SamplingFunctions, Figsize, TightLayout


class PuzzlesManager(AbstractManager):
    """Data Manager for the Puzzles Dataset."""

    Bound = Dict[str, Tuple[int, int]]

    # noinspection PyMissingOrEmptyDocstring
    @staticmethod
    def load_data(filepath: str, extrapolation: bool) -> AbstractManager.Data:
        df = pd.read_csv(filepath)  # raw dataframe with special preprocessing needs
        for col in df.columns:
            if col not in ['label', 'split']:
                df[col] = df[col].map(lambda l: l.strip('[]').split(';')).map(lambda l: [float(v.strip()) for v in l])
        # df = df.apply(lambda s: s.map(lambda l: np.mean(l)))
        x = pd.DataFrame()
        x['word_count'] = df['word_count'].map(lambda l: np.mean(l))
        x['star_rating'] = df['star_rating'].map(lambda l: np.mean(l))
        x['num_reviews'] = df['star_rating'].map(lambda l: len(l))
        x['label'] = df['label']
        x['split'] = df['split']
        df = x.copy()
        # split train/test
        if extrapolation:
            splits = split_dataset(df, val_size=0.2, extrapolation=0.2)
        else:
            splits = {
                'train': df[np.in1d(df['split'], ['train', 'validation'])],
                'test': df[df['split'] == 'test']
            }
        return {split: data.drop(columns='split') for split, data in splits.items()}

    def __init__(self, filepath: str, full_features: bool = False, full_grid: bool = False, extrapolation: bool = False,
                 x_scaling: Methods = 'std', y_scaling: Methods = 'norm', bound: Optional[Bound] = None):
        self.bound = {'word_count': (0, 230), 'star_rating': (0, 5), 'num_reviews': (0, 70)} if bound is None else bound
        grid = None
        if full_features:
            assert full_grid is False, "'full_grid' is not supported with 'full_features'"
        elif full_grid:
            wc, sr, nr = np.meshgrid(
                np.linspace(self.bound['word_count'][0], self.bound['word_count'][1], 20),
                np.linspace(self.bound['star_rating'][0], self.bound['star_rating'][1], 20),
                np.linspace(self.bound['num_reviews'][0], self.bound['num_reviews'][1], 20)
            )
            grid = pd.DataFrame({'word_count': wc.flatten(), 'star_rating': sr.flatten(), 'num_reviews': nr.flatten()})
        super(PuzzlesManager, self).__init__(
            directions={'word_count': -1, 'star_rating': 1, 'num_reviews': 1},
            stratify=False,
            x_scaling=x_scaling,
            y_scaling=y_scaling,
            label='label',
            loss=mean_squared_error,
            loss_name='mse',
            metric=r2_score,
            metric_name='r2',
            data_kwargs=dict(figsize=(12, 10)),
            augmented_kwargs=dict(figsize=(14, 4), tight_layout=True),
            summary_kwargs=dict(figsize=(14, 4), tight_layout=True, res=5),
            grid_kwargs=dict(num_augmented=30),
            grid=grid,
            filepath=filepath,
            extrapolation=extrapolation
        )

    def _get_sampling_functions(self, rng: Rng, num_augmented: Augmented = (3, 4, 8)) -> SamplingFunctions:
        if isinstance(num_augmented, int):
            num_augmented = [num_augmented] * 3
        elif isinstance(num_augmented, dict):
            num_augmented = [num_augmented[k] for k in ['word_count', 'star_rating', 'num_reviews']]
        elif isinstance(num_augmented, tuple):
            num_augmented = list(num_augmented)
        b = self.bound
        return {
            'word_count': (num_augmented[0], lambda s: rng.uniform(b['word_count'][0], b['word_count'][1], size=s)),
            'star_rating': (num_augmented[0], lambda s: rng.uniform(b['star_rating'][0], b['star_rating'][1], size=s)),
            'num_reviews': (num_augmented[0], lambda s: rng.uniform(b['num_reviews'][0], b['num_reviews'][1], size=s))
        }

    def _data_plot(self, figsize: Figsize, tight_layout: TightLayout, **kwargs):
        dfs, info = [], []
        for key, (x, y) in kwargs.items():
            df = pd.concat((x, y), axis=1)
            df['Key'] = key.capitalize()
            dfs.append(df)
        w, h = figsize
        sns.pairplot(data=pd.concat(dfs), hue='Key', plot_kws={'alpha': 0.7}, height=h / 4, aspect=w / h)

    def _summary_plot(self, model, figsize: Figsize, tight_layout: TightLayout, **kwargs):
        res = kwargs.pop('res')
        fig, axes = plt.subplots(1, 3, sharey='all', tight_layout=tight_layout, figsize=figsize)
        wc, sr, nr = np.meshgrid(
            np.linspace(self.bound['word_count'][0], self.bound['word_count'][1], res),
            np.linspace(self.bound['star_rating'][0], self.bound['star_rating'][1], res),
            np.linspace(self.bound['num_reviews'][0], self.bound['num_reviews'][1], res)
        )
        grid = pd.DataFrame({'word_count': wc.flatten(), 'star_rating': sr.flatten(), 'num_reviews': nr.flatten()})
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
