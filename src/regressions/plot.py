import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.util.preprocessing import Scaler


def plot_cars(scalers=None, figsize=(14, 4), tight_layout=True, **kwargs):
    info = []
    x_scaler, y_scaler = (Scaler.get_default(1), Scaler.get_default(1)) if scalers is None else scalers
    _, axes = plt.subplots(1, len(kwargs), sharex='all', sharey='all', figsize=figsize, tight_layout=tight_layout)
    for ax, (title, (x, y)) in zip(axes, kwargs.items()):
        info.append(f'{len(x)} {title} samples')
        x = x_scaler.invert(x['price'])
        y = y_scaler.invert(y)
        sns.scatterplot(x=x, y=y, ax=ax).set(xlabel='price', ylabel='sales', title=title.capitalize())
    print(', '.join(info))


def plot_cars_augmented(x, y, x_scaler=None, figsize=(14, 4)):
    scl = Scaler.get_default(1) if x_scaler is None else x_scaler
    aug = scl.invert(x)
    aug['Augmented'] = np.isnan(y)
    plt.figure(figsize=figsize)
    sns.histplot(data=aug, x='price', hue='Augmented')


def plot_puzzles(scalers=None, figsize=(12, 10), **kwargs):
    x_scaler, y_scaler = (Scaler.get_default(3), Scaler.get_default(3)) if scalers is None else scalers
    dfs, info = [], []
    for key, (x, y) in kwargs.items():
        df = pd.concat((x_scaler.invert(x), y_scaler.invert(y)), axis=1)
        df['Key'] = key.capitalize()
        dfs.append(df)
        info.append(f'{len(x)} {key} samples')
    dfs = pd.concat(dfs)
    print(', '.join(info))
    w, h = figsize
    sns.pairplot(dfs, hue='Key', plot_kws={'alpha': 0.7}, height=h / 4, aspect=w / h)


def plot_puzzles_augmented(x, y, x_scaler=None, figsize=(14, 4), tight_layout=True):
    scl = Scaler.get_default(3) if x_scaler is None else x_scaler
    aug = scl.invert(x)
    aug['Augmented'] = np.isnan(y)
    _, axes = plt.subplots(1, 3, sharey='all', figsize=figsize, tight_layout=tight_layout)
    for ax, feature in zip(axes, ['word_count', 'star_rating', 'num_reviews']):
        sns.histplot(data=aug, x=feature, hue='Augmented', ax=ax)
