import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.util.augmentation import get_monotonicities_list, compute_numeric_monotonicities, augment_data
from src.util.model import violations_summary, metrics_summary
from src.util.preprocessing import Scaler


class Dataset:
    def __init__(self, x_columns, x_scaling, y_column, y_scaling, metric, res, directions=None):
        self.x_columns = x_columns
        self.x_scaling = x_scaling
        self.y_column = y_column
        self.y_scaling = y_scaling
        self.directions = directions
        self.metric = metric
        self.grid = pd.DataFrame.from_dict({col: np.linspace(lb, ub, res) for col, (lb, ub) in x_columns.items()})
        self.monotonicities = get_monotonicities_list(
            data=self.grid,
            label=None,
            kind='all',
            errors='ignore',
            compute_monotonicities=self.compute_monotonicities
        )

    def compute_monotonicities(self, samples, references):
        return compute_numeric_monotonicities(samples, references, directions=self.directions)

    def get_scalers(self, x, y):
        x_scaler = Scaler(self.x_scaling).fit(x)
        y_scaler = Scaler(self.y_scaling).fit(y)
        return x_scaler, y_scaler

    def load_data(self, **kwargs):
        pass

    def get_augmented_data(self, x, y, num_augmented=15, seed=0):
        rng = np.random.default_rng(seed=seed)
        x_aug, y_aug = augment_data(x=x, y=y, compute_monotonicities=self.compute_monotonicities, sampling_functions={
            col: (num_augmented, lambda s: rng.uniform(lb, ub, size=s)) for col, (lb, ub) in self.x_columns.items()
        })
        mask = ~np.isnan(y_aug[self.y_column])
        return (x_aug, y_aug), self.get_scalers(x_aug, y_aug[self.y_column][mask])

    def plot_data(self, figsize=(14, 4), tight_layout=True, **kwargs):
        pass

    def plot_augmented(self, x, y, figsize=(14, 4), tight_layout=True):
        aug = x.copy()
        aug['Augmented'] = np.isnan(y[self.y_column])
        plt.figure(figsize=figsize)
        if len(self.x_columns) == 1:
            sns.histplot(data=aug, x=list(self.x_columns.keys())[0], hue='Augmented')
        else:
            _, axes = plt.subplots(1, len(self.x_columns), sharey='all', figsize=figsize, tight_layout=tight_layout)
            for ax, feature in zip(axes, list(self.x_columns.keys())):
                sns.histplot(data=aug, x=feature, hue='Augmented', ax=ax)
        plt.show()

    def evaluation_summary(self, model, **kwargs):
        # compute metrics on kwargs and plot data points
        print(violations_summary(model=model, grid=self.grid, monotonicities=self.monotonicities))
        print(metrics_summary(model=model, metric=self.metric, **kwargs))
