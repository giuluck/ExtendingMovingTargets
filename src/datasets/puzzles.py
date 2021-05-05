import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from src.datasets.dataset import Dataset
from src.util.preprocessing import split_dataset
from util.augmentation import compute_numeric_monotonicities
from util.plot import ColorFader


class Puzzles(Dataset):
    def __init__(self, x_scaling='std', y_scaling='norm', res=20, bound=None):
        self.bound = {'word_count': (0, 180), 'star_rating': (0, 5), 'num_reviews': (0, 60)} if bound is None else bound
        wc, nr, sr = np.meshgrid(
            np.linspace(self.bound['word_count'][0], self.bound['word_count'][1], res),
            np.linspace(self.bound['star_rating'][0], self.bound['star_rating'][1], res),
            np.linspace(self.bound['num_reviews'][0], self.bound['num_reviews'][1], res)
        )
        super(Puzzles, self).__init__(
            x_columns=['word_count', 'star_rating', 'num_reviews'],
            x_scaling=x_scaling,
            y_column='label',
            y_scaling=y_scaling,
            metric=r2_score,
            grid=pd.DataFrame({'word_count': wc.flatten(), 'star_rating': sr.flatten(), 'num_reviews': nr.flatten()}),
            data_kwargs=dict(figsize=(12, 10)),
            augmented_kwargs=dict(figsize=(14, 4), tight_layout=True),
            summary_kwargs=dict(figsize=(14, 4), tight_layout=True, res=5)
        )

    def compute_monotonicities(self, samples, references):
        return compute_numeric_monotonicities(samples, references, directions=[-1, 1, 1])

    def _load_splits(self, filepath, extrapolation=False):
        df = pd.read_csv(filepath)
        for col in df.columns:
            if col not in ['label', 'split']:
                df[col] = df[col].map(lambda l: l.strip('[]').split(';')).map(lambda l: [float(v.strip()) for v in l])
        x = pd.DataFrame()
        x['word_count'] = df['word_count'].map(lambda l: np.mean(l))
        x['star_rating'] = df['star_rating'].map(lambda l: np.mean(l))
        x['num_reviews'] = df['star_rating'].map(lambda l: len(l))
        y = df['label']
        # split data
        if extrapolation:
            return split_dataset(x, y, val_size=0.2, extrapolation=0.2)
        else:
            return {s: (x[df['split'] == s], y[df['split'] == s]) for s in ['train', 'validation', 'test']}

    def _get_sampling_functions(self, num_augmented, rng):
        if isinstance(num_augmented, int):
            num_augmented = [num_augmented] * 3
        elif isinstance(num_augmented, dict):
            num_augmented = [num_augmented[k] for k in ['word_count', 'star_rating', 'num_reviews']]
        b = self.bound
        return {
            'word_count': (num_augmented[0], lambda s: rng.uniform(b['word_count'][0], b['word_count'][1], size=s)),
            'star_rating': (num_augmented[0], lambda s: rng.uniform(b['star_rating'][0], b['star_rating'][1], size=s)),
            'num_reviews': (num_augmented[0], lambda s: rng.uniform(b['num_reviews'][0], b['num_reviews'][1], size=s))
        }

    def _data_plot(self, figsize, **kwargs):
        dfs, info = [], []
        for key, (x, y) in kwargs.items():
            df = pd.concat((x, y), axis=1)
            df['Key'] = key.capitalize()
            dfs.append(df)
        w, h = figsize
        sns.pairplot(data=pd.concat(dfs), hue='Key', plot_kws={'alpha': 0.7}, height=h / 4, aspect=w / h)

    # noinspection PyMethodOverriding
    def _summary_plot(self, model, res, figsize, tight_layout, **kwargs):
        fig, axes = plt.subplots(1, 3, sharey='all', tight_layout=tight_layout, figsize=figsize)
        wc, nr, sr = np.meshgrid(
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
            fader = ColorFader('black', 'magenta', 'cyan', 'yellow', bounds=(li, lj, ui, uj))
            for (i, j), group in grid.groupby([fi, fj]):
                label = f'{fi}: {i:.0f}, {fj}: {j:.0f}' if (i in [li, ui] and j in [lj, uj]) else None
                sns.lineplot(data=group, x=feat, y='pred', color=fader(i, j), alpha=0.6, label=label, ax=ax)
        fig.suptitle('Estimated Functions')
        plt.show()
