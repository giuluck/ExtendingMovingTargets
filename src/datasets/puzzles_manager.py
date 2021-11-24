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
    """Type for summary plot bounds. It is a dictionary that associates to each monotonic feature a tuple of integers
    representing the min/max y bounds of that feature."""

    @staticmethod
    def load_data(filepath: str, full_features: bool, extrapolation: bool) -> AbstractManager.Data:
        """Loads the dataset.

        :param filepath:
            The dataset filepath.

        :param full_features:
            If True considers all the dataset features, otherwise considers only the monotonic ones.

        :param extrapolation:
            Whether to consider a test set with border samples or with random samples (interpolation).

        :return:
            A dictionary of dataframes representing the train and test sets, respectively.
        """
        def _process(series: pd.Series) -> pd.Series:
            """Converts the each string entry of a series into the respective list and takes the average value.

            :param series:
                An input column of the puzzles dataset.

            :return:
                A series of average values for each list entry.
            """
            # apart from these three columns,
            if series.name in ['label', 'split', 'num_reviews']:
                return series
            else:
                return series.map(lambda l: np.mean([float(v.strip()) for v in l.strip('[]').split(';')]))

        columns = ['star_rating', 'word_count', 'num_reviews']
        df = pd.read_csv(filepath)
        df['num_reviews'] = df['is_amazon'].map(lambda l: len(l.split(';'))).astype(float)
        df = df.apply(_process)
        df = df.dropna() if full_features else df[columns + ['label', 'split']].dropna()
        if extrapolation:
            extrapolation = {c: 0.2 for c in columns}
            splits = split_dataset(df, extrapolation=extrapolation, val_size=0.0)
        else:
            splits = {'train': df[np.in1d(df['split'], ['train', 'validation'])], 'test': df[df['split'] == 'test']}
        return {split: data.drop(columns='split') for split, data in splits.items()}

    def __init__(self, filepath: str, full_features: bool = False, full_grid: bool = False, extrapolation: bool = False,
                 grid_augmented: int = 35, grid_ground: Optional[int] = None, x_scaling: Methods = 'std',
                 y_scaling: Methods = 'norm', bound: Optional[Bound] = None):
        """
        :param filepath:
            The cars dataset filepath.

        :param full_features:
            Whether or not to use all the input features.

        :param full_grid:
            Whether or not to evaluate the results on an explicit full grid. This option is possible only if
            full_features is set to false, since there is no way to create an explicit grid on the full set of features.

        :param extrapolation:
            Whether or not to test on extrapolation instead of interpolation.

        :param grid_augmented:
            Number of augmented features for grid_kwargs. This will not have any effect if an explicit grid is passed.

        :param grid_ground:
            Number of ground samples for grid_kwargs. This will not have any effect if an explicit grid is passed.

        :param x_scaling:
            Scaling methods for the input data.

        :param y_scaling:
            Scaling methods for the output data.

        :param bound:
            The y bounds for the summary plot.
        """

        self.bound = {'word_count': (0, 230), 'star_rating': (0, 5), 'num_reviews': (0, 70)} if bound is None else bound
        """A dictionary that associates to each monotonic feature its bounds for the summary plot."""

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
            grid_kwargs=dict(num_augmented=grid_augmented, num_ground=grid_ground),
            grid=grid,
            filepath=filepath,
            full_features=full_features,
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

    def _data_plot(self, figsize: Figsize, tight_layout: TightLayout, **additional_kwargs):
        dfs, info = [], []
        for key, (x, y) in additional_kwargs.items():
            df = pd.concat((x, y), axis=1)
            df['Key'] = key.capitalize()
            dfs.append(df)
        w, h = figsize
        sns.pairplot(data=pd.concat(dfs), hue='Key', plot_kws={'alpha': 0.7}, height=h / 4, aspect=w / h)

    def _summary_plot(self, model, figsize: Figsize, tight_layout: TightLayout, **additional_kwargs):
        res = additional_kwargs.pop('res')
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
