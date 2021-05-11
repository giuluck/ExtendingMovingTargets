import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from src.datasets.data_manager import DataManager
from src.util.preprocessing import split_dataset
from src.util.augmentation import compute_numeric_monotonicities


class CarsManager(DataManager):
    def __init__(self, filepath, x_scaling='std', y_scaling='norm', bound=(0, 100), res=700):
        self.filepath = filepath
        self.bound = bound
        super(CarsManager, self).__init__(
            x_columns=['price'],
            x_scaling=x_scaling,
            y_column='sales',
            y_scaling=y_scaling,
            metric=r2_score,
            metric_name='r2',
            grid=pd.DataFrame.from_dict({'price': np.linspace(self.bound[0], self.bound[1], res)}),
            data_kwargs=dict(figsize=(14, 4), tight_layout=True),
            augmented_kwargs=dict(figsize=(10, 4)),
            summary_kwargs=dict(figsize=(10, 4), res=100, ylim=(-5, 125))
        )

    def compute_monotonicities(self, samples, references, eps=1e-5):
        return compute_numeric_monotonicities(samples, references, directions=[-1], eps=eps)

    def _load_splits(self, extrapolation=False):
        # preprocess data
        df = pd.read_csv(self.filepath).rename(
            columns={'Price in thousands': 'price', 'Sales in thousands': 'sales'})
        df = df[['price', 'sales']].replace({'.': np.nan}).dropna().astype('float')
        # split data
        if extrapolation:
            return split_dataset(df[['price']], df['sales'], extrapolation=0.2, val_size=0.2, random_state=0)
        else:
            return split_dataset(df[['price']], df['sales'], extrapolation=None, test_size=0.2, random_state=0)

    def _get_sampling_functions(self, num_augmented, rng):
        return {'price': (num_augmented, lambda s: rng.uniform(self.bound[0], self.bound[1], size=s))}

    def _data_plot(self, figsize, tight_layout, **kwargs):
        _, axes = plt.subplots(1, len(kwargs), sharex='all', sharey='all', figsize=figsize, tight_layout=tight_layout)
        for ax, (title, (x, y)) in zip(axes, kwargs.items()):
            sns.scatterplot(x=x['price'], y=y, ax=ax).set(xlabel='price', ylabel='sales', title=title.capitalize())

    # noinspection PyMethodOverriding
    def _augmented_plot(self, aug, figsize, **kwargs):
        plt.figure(figsize=figsize)
        sns.histplot(data=aug, x='price', hue='Augmented')

    # noinspection PyMethodOverriding
    def _summary_plot(self, model, res, ylim, figsize, **kwargs):
        plt.figure(figsize=figsize)
        for title, (x, y) in kwargs.items():
            sns.scatterplot(x=x['price'], y=y, alpha=0.25, sizes=0.25, label=title.capitalize())
        x = np.linspace(self.bound[0], self.bound[1], res)
        y = model.predict(x.reshape(-1, 1)).flatten()
        sns.lineplot(x=x, y=y, color='black').set(xlabel='price', ylabel='sales', title='Estimated Function')
        plt.xlim(self.bound)
        plt.ylim(ylim)
