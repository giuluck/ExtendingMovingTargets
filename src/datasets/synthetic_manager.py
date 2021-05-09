import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from src.datasets.data_manager import DataManager
from src.util.preprocessing import split_dataset
from src.util.augmentation import compute_numeric_monotonicities
from src.util.plot import ColorFader


class SyntheticManager(DataManager):
    @staticmethod
    def function(a, b):
        a = a ** 3
        b = np.sin(np.pi * (b - 0.01)) ** 2 + 1
        return a / b + b

    def __init__(self, noise=0.0, x_scaling='std', y_scaling='norm', res=80):
        self.noise = noise
        a, b = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
        super(SyntheticManager, self).__init__(
            x_columns=['a', 'b'],
            x_scaling=x_scaling,
            y_column='label',
            y_scaling=y_scaling,
            metric=r2_score,
            grid=pd.DataFrame({'a': a.flatten(), 'b': b.flatten()}),
            data_kwargs=dict(figsize=(12, 10), tight_layout=True),
            augmented_kwargs=dict(figsize=(10, 4), tight_layout=True),
            summary_kwargs=dict(figsize=(14, 8), tight_layout=True, res=50)
        )

    def compute_monotonicities(self, samples, references, eps=1e-5):
        return compute_numeric_monotonicities(samples, references, directions=[1, 0], eps=eps)

    def _load_splits(self, extrapolation=False):
        # generate and split data
        rng = np.random.default_rng(seed=0)
        if extrapolation:
            df = pd.DataFrame.from_dict({
                'a': rng.uniform(low=-1, high=1, size=700),
                'b': rng.uniform(low=-1, high=1, size=700)
            })
            splits = split_dataset(df, extrapolation={'a': 0.7}, val_size=0.25, random_state=0)
        else:
            df = [
                {'a': rng.normal(scale=0.3, size=150).clip(min=-1, max=1),
                 'b': rng.uniform(low=-1, high=1, size=150)},
                {'a': rng.normal(scale=0.3, size=50).clip(min=-1, max=1),
                 'b': rng.uniform(low=-1, high=1, size=50)},
                {'a': rng.uniform(low=-1, high=1, size=500), 'b': rng.uniform(low=-1, high=1, size=500)}
            ]
            splits = {s: pd.DataFrame.from_dict(x) for s, x in zip(['train', 'validation', 'test'], df)}
        # assign y values
        outputs = {}
        for s, x in splits.items():
            y = pd.Series(SyntheticManager.function(x['a'], x['b']), name='label') + rng.normal(scale=self.noise, size=len(x))
            outputs[s] = (x, y)
        return outputs

    def _get_sampling_functions(self, num_augmented, rng):
        return {'a': (num_augmented, lambda s: rng.uniform(-1, 1, size=s))}

    def _data_plot(self, figsize, tight_layout, **kwargs):
        _, ax = plt.subplots(len(kwargs), 3, sharex='row', sharey='row', figsize=figsize, tight_layout=tight_layout)
        # hue/size bounds
        ybn = np.concatenate([[y.min(), y.max()] for _, y in kwargs.values()])
        abn, bbn, ybn = (-1, 1), (-1, 1), (np.min(ybn), np.max(ybn))
        # plots
        for i, (title, (x, y)) in enumerate(kwargs.items()):
            a, b = x['a'], x['b']
            ax[0, i].set(title=title.capitalize())
            sns.scatterplot(x=a, y=y, hue=b, hue_norm=bbn, size=b, size_norm=bbn, ax=ax[0, i], legend=False)
            ax[0, i].legend([f'b {bbn}'], markerscale=0, handlelength=0)
            sns.scatterplot(x=b, y=y, hue=a, hue_norm=abn, size=a, size_norm=abn, ax=ax[1, i], legend=False)
            ax[1, i].legend([f'a {abn}'], markerscale=0, handlelength=0)
            sns.scatterplot(x=a, y=b, hue=y, hue_norm=ybn, size=y, size_norm=ybn, ax=ax[2, i], legend=False)
            ax[2, i].legend([f'label ({ybn[0]:.0f}, {ybn[1]:.0f})'], markerscale=0, handlelength=0)

    # noinspection PyMethodOverriding
    def _summary_plot(self, model, res, figsize, tight_layout, **kwargs):
        a, b = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
        grid = pd.DataFrame.from_dict({'a': a.flatten(), 'b': b.flatten()})
        grid['pred'] = model.predict(grid)
        grid['label'] = SyntheticManager.function(grid['a'], grid['b'])
        fader = ColorFader('red', 'blue', bounds=(-1, 1))
        _, axes = plt.subplots(2, 3, figsize=figsize, tight_layout=tight_layout)
        for ax, (title, y) in zip(axes, {'Ground Truth': 'label', 'Estimated Function': 'pred'}.items()):
            # plot bivariate function
            z = grid[y].values.reshape(res, res)
            ax[0].pcolor(a, b, z, shading='auto', cmap='viridis', vmin=grid['label'].min(), vmax=grid['label'].max())
            # plot first feature (with title as it is the central plot)
            for idx, group in grid.groupby('b'):
                label = f'b = {idx:.0f}' if idx in [-1, 1] else None
                sns.lineplot(data=group, x='a', y=y, color=fader(idx), alpha=0.4, label=label, ax=ax[1])
            # plot second feature
            for idx, group in grid.groupby('a'):
                label = f'a = {idx:.0f}' if idx in [-1, 1] else None
                sns.lineplot(data=group, x='b', y=y, color=fader(idx), alpha=0.4, label=label, ax=ax[2])
