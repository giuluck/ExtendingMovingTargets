import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from src.datasets.dataset import Dataset
from src.util.preprocessing import split_dataset


class Cars(Dataset):
    def __init__(self, x_scaling='std', y_scaling='norm', res=700):
        super(Cars, self).__init__(
            x_columns={'price': (0, 100)},
            x_scaling=x_scaling,
            y_column='sales',
            y_scaling=y_scaling,
            metric=r2_score,
            res=res,
            directions=[-1]
        )

    def load_data(self, filepath, extrapolation=False):
        # preprocess data
        df = pd.read_csv(filepath).rename(
            columns={'Price in thousands': 'price', 'Sales in thousands': 'sales'})
        df = df[['price', 'sales']].replace({'.': np.nan}).dropna().astype('float')
        # split data
        if extrapolation:
            splits = split_dataset(df[['price']], df['sales'], extrapolation=0.2, val_size=0.2, random_state=0)
        else:
            splits = split_dataset(df[['price']], df['sales'], extrapolation=None, test_size=0.2, random_state=0)
        return splits, self.get_scalers(splits['train'][0], splits['train'][1])

    def plot_data(self, figsize=(14, 4), tight_layout=True, **kwargs):
        info = []
        _, axes = plt.subplots(1, len(kwargs), sharex='all', sharey='all', figsize=figsize, tight_layout=tight_layout)
        for ax, (title, (x, y)) in zip(axes, kwargs.items()):
            info.append(f'{len(x)} {title} samples')
            sns.scatterplot(x=x['price'], y=y, ax=ax).set(xlabel='price', ylabel='sales', title=title.capitalize())
        print(', '.join(info))
        plt.show()

    def evaluation_summary(self, model, res=100, ylim=(-5, 125), figsize=(10, 4), **kwargs):
        super(Cars, self).evaluation_summary(model=model, **kwargs)
        plt.figure(figsize=figsize)
        for title, (x, y) in kwargs.items():
            sns.scatterplot(x=x['price'], y=y, alpha=0.25, sizes=0.25, label=title.capitalize())
        x_lower, x_upper = self.x_columns['price']
        x = np.linspace(x_lower, x_upper, res)
        y = model.predict(x.reshape(-1, 1)).flatten()
        sns.lineplot(x=x, y=y, color='black').set(xlabel='price', ylabel='sales', title='Estimated Function')
        plt.xlim((x_lower, x_upper))
        plt.ylim(ylim)
        plt.show()
