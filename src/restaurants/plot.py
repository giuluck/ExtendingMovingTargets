import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_histograms(figsize=(10, 8), tight_layout=True, **kwargs):
    if len(kwargs) > 0:
        _, axes = plt.subplots(len(kwargs), 3, sharex='col', sharey='row', figsize=figsize, tight_layout=tight_layout)
        axes = axes.reshape((-1, 3))
        for i, (title, x) in enumerate(kwargs.items()):
            sns.histplot(x=x['avg_rating'], ax=axes[i, 0]).set(ylabel=title.capitalize())
            sns.histplot(x=x['num_reviews'], ax=axes[i, 1])
            sns.histplot(x=x[['D', 'DD', 'DDD', 'DDDD']].idxmax(axis=1), ax=axes[i, 2])
    plt.show()


def plot_distributions(figsize=(10, 8), tight_layout=True, **kwargs):
    _, axes = plt.subplots(len(kwargs), 3, sharex='col', figsize=figsize, tight_layout=tight_layout)
    axes = axes.reshape((-1, 3))
    for i, (title, (x, y)) in enumerate(kwargs.items()):
        sns.kdeplot(x=x['avg_rating'], hue=y, ax=axes[i, 0]).set(ylabel=title.capitalize())
        sns.kdeplot(x=x['num_reviews'], hue=y, ax=axes[i, 1])
        sns.countplot(x=x[['D', 'DD', 'DDD', 'DDDD']].idxmax(axis=1), hue=y, ax=axes[i, 2])
    plt.show()


def plot_conclusions(models, figsize=(10, 10), res=100, orient_columns=True):
    avg_ratings, num_reviews = np.meshgrid(np.linspace(1, 5, num=res), np.linspace(0, 200, num=res))
    rows, cols = (4, len(models)) if orient_columns else (len(models), 4)
    _, axes = plt.subplots(rows, cols, figsize=figsize, sharex='all', sharey='all', tight_layout=True)
    axes = axes.reshape((rows, cols))

    for i, rating in enumerate(['D', 'DD', 'DDD', 'DDDD']):
        for j, (title, model) in enumerate(models.items()):
            ctr = model.ctr_estimate(avg_ratings.flatten(), num_reviews.flatten(), [rating] * (res ** 2))
            if orient_columns:
                axes[i, j].pcolor(avg_ratings, num_reviews, ctr.reshape(res, res), shading='auto', vmin=0, vmax=1)
                axes[i, j].set(title=title if i == 0 else None, xlabel=None, ylabel=rating if j == 0 else None)
            else:
                axes[j, i].pcolor(avg_ratings, num_reviews, ctr.reshape(res, res), shading='auto', vmin=0, vmax=1)
                axes[j, i].set(title=rating if j == 0 else None, xlabel=None, ylabel=title if i == 0 else None)

    plt.show()
