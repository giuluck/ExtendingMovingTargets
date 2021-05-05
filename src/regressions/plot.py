import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.util.preprocessing import Scaler


def plot_synthetic(scalers=None, figsize=(12, 10), tight_layout=True, **kwargs):
    x_scaler, y_scaler = (Scaler.get_default(2), Scaler.get_default(1)) if scalers is None else scalers
    _, ax = plt.subplots(len(kwargs), 3, sharex='row', sharey='row', figsize=figsize, tight_layout=tight_layout)
    # hue/size bounds
    ybn = y_scaler.inverse_transform(np.concatenate([[y.min(), y.max()] for _, y in kwargs.values()]))
    ybn = (ybn.min(), ybn.max())
    abn, bbn = (-1, 1), (-1, 1)
    # plots
    info = []
    for i, (title, (x, y)) in enumerate(kwargs.items()):
        x, y = x_scaler.inverse_transform(x), y_scaler.inverse_transform(y)
        ax[0, i].set(title=title.capitalize())
        sns.scatterplot(x=x['a'], y=y, hue=x['b'], hue_norm=bbn, size=x['b'], size_norm=bbn, ax=ax[0, i], legend=False)
        ax[0, i].legend([f'b {bbn}'], markerscale=0, handlelength=0)
        sns.scatterplot(x=x['b'], y=y, hue=x['a'], hue_norm=abn, size=x['a'], size_norm=abn, ax=ax[1, i], legend=False)
        ax[1, i].legend([f'a {abn}'], markerscale=0, handlelength=0)
        sns.scatterplot(x=x['a'], y=x['b'], hue=y, hue_norm=ybn, size=y, size_norm=ybn, ax=ax[2, i], legend=False)
        ax[2, i].legend([f'label ({ybn[0]:.0f}, {ybn[1]:.0f})'], markerscale=0, handlelength=0)
        info.append(f'{len(x)} ({title} samples)')
    print(', '.join(info))
    plt.show()


def plot_synthetic_augmented(x, y, x_scaler=None, figsize=(10, 4), tight_layout=True):
    scl = Scaler.get_default(2) if x_scaler is None else x_scaler
    aug = scl.inverse_transform(x)
    aug['Augmented'] = np.isnan(y)
    _, axes = plt.subplots(1, 2, sharey='all', figsize=figsize, tight_layout=tight_layout)
    for ax, feature in zip(axes, ['a', 'b']):
        sns.histplot(data=aug, x=feature, hue='Augmented', ax=ax)
    plt.show()


def plot_puzzles(scalers=None, figsize=(12, 10), **kwargs):
    x_scaler, y_scaler = (Scaler.get_default(3), Scaler.get_default(1)) if scalers is None else scalers
    dfs, info = [], []
    for key, (x, y) in kwargs.items():
        df = pd.concat((x_scaler.inverse_transform(x), y_scaler.inverse_transform(y)), axis=1)
        df['Key'] = key.capitalize()
        dfs.append(df)
        info.append(f'{len(x)} {key} samples')
    dfs = pd.concat(dfs)
    print(', '.join(info))
    w, h = figsize
    sns.pairplot(dfs, hue='Key', plot_kws={'alpha': 0.7}, height=h / 4, aspect=w / h)
    plt.show()


def plot_puzzles_augmented(x, y, x_scaler=None, figsize=(14, 4), tight_layout=True):
    scl = Scaler.get_default(3) if x_scaler is None else x_scaler
    aug = scl.inverse_transform(x)
    aug['Augmented'] = np.isnan(y)
    _, axes = plt.subplots(1, 3, sharey='all', figsize=figsize, tight_layout=tight_layout)
    for ax, feature in zip(axes, ['word_count', 'star_rating', 'num_reviews']):
        sns.histplot(data=aug, x=feature, hue='Augmented', ax=ax)
    plt.show()
