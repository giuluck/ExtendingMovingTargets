import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.util.augmentation import get_monotonicities_list, augment_data
from src.util.model import violations_summary, metrics_summary
from src.util.preprocessing import Scaler


class Dataset:
    def __init__(self, x_columns, x_scaling, y_column, y_scaling, metric, grid,
                 data_kwargs, augmented_kwargs, summary_kwargs):
        self.x_columns = x_columns
        self.x_scaling = x_scaling
        self.y_column = y_column
        self.y_scaling = y_scaling
        self.metric = metric
        self.grid = grid
        self.monotonicities = get_monotonicities_list(
            data=self.grid,
            label=None,
            kind='all',
            errors='ignore',
            compute_monotonicities=self.compute_monotonicities
        )
        self.data_kwargs = data_kwargs
        self.augmented_kwargs = augmented_kwargs
        self.summary_kwargs = summary_kwargs

    def _load_splits(self, **kwargs):
        raise NotImplementedError("please implement method '_load_splits'")

    def _get_sampling_functions(self, num_augmented, rng):
        raise NotImplementedError("please implement method '_get_sampling_functions'")

    def _data_plot(self, **kwargs):
        raise NotImplementedError("please implement method '_data_plot'")

    def _augmented_plot(self, aug, **kwargs):
        figsize, tight_layout = kwargs.get('figsize'), kwargs.get('tight_layout')
        _, axes = plt.subplots(1, len(self.x_columns), sharey='all', figsize=figsize, tight_layout=tight_layout)
        for ax, feature in zip(axes, self.x_columns):
            sns.histplot(data=aug, x=feature, hue='Augmented', ax=ax)

    def _summary_plot(self, model, **kwargs):
        raise NotImplementedError("please implement method '_summary_plot'")

    def compute_monotonicities(self, samples, references):
        raise NotImplementedError("please implement method '_compute_monotonicities'")

    def get_scalers(self, x, y):
        x_scaler = Scaler(self.x_scaling).fit(x)
        y_scaler = Scaler(self.y_scaling).fit(y)
        return x_scaler, y_scaler

    def load_data(self, **kwargs):
        splits = self._load_splits(**kwargs)
        return splits, self.get_scalers(splits['train'][0], splits['train'][1])

    def get_augmented_data(self, x, y, num_augmented=15, num_random=0, num_ground=None, seed=0):
        rng = np.random.default_rng(seed=seed)
        # handle input samples reduction
        if num_ground is not None:
            x = x.head(num_ground)
            y = y.head(num_ground)
        # add random unsupervised samples to fill the data space
        if num_random > 0:
            random_values = {}
            sampling_functions = self._get_sampling_functions(num_augmented=num_random, rng=rng)
            for col in x.columns:
                # if there is an explicit sampling strategy use it, otherwise sample original data
                n, f = sampling_functions.get(col, (num_random, lambda s: rng.choice(x[col], size=s)))
                random_values[col] = f(n)
            x = pd.concat((x, pd.DataFrame.from_dict(random_values)), ignore_index=True)
            y = pd.concat((y, pd.Series([np.nan] * num_random, name=y.name)), ignore_index=True)
        # augment data
        x_aug, y_aug = augment_data(
            x=x,
            y=y,
            compute_monotonicities=self.compute_monotonicities,
            sampling_functions=self._get_sampling_functions(num_augmented=num_augmented, rng=rng)
        )
        mask = ~np.isnan(y_aug[self.y_column])
        return (x_aug, y_aug), self.get_scalers(x=x_aug, y=y_aug[self.y_column][mask])

    def plot_data(self, **kwargs):
        # print general info about data
        info = [f'{len(x)} {title} samples' for title, (x, _) in kwargs.items()]
        print(', '.join(info))
        # plot data
        kw = self.data_kwargs.copy()
        kw.update(kwargs)
        plt.figure(figsize=kw['figsize'])
        self._data_plot(**kw)
        plt.show()

    def plot_augmented(self, x, y, **kwargs):
        # retrieve augmented data
        aug = x.copy()
        aug['Augmented'] = np.isnan(y[self.y_column])
        # plot augmented data
        kw = self.augmented_kwargs.copy()
        kw.update(kwargs)
        plt.figure(figsize=kw['figsize'])
        self._augmented_plot(aug=aug, **kw)
        plt.show()

    def evaluation_summary(self, model, **kwargs):
        # compute metrics on kwargs
        print(violations_summary(model=model, grid=self.grid, monotonicities=self.monotonicities))
        print(metrics_summary(model=model, metric=self.metric, **kwargs))
        # plot summary
        kw = self.summary_kwargs.copy()
        kw.update(kwargs)
        plt.figure(figsize=kw['figsize'])
        self._summary_plot(model=model, **kw)
        plt.show()
