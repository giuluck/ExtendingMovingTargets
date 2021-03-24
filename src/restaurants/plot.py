import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_ctr(ctr_estimate, title=None, figsize=(14, 3), res=100):
    def color_bar():
        # noinspection PyUnresolvedReferences
        bar = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1, True), cmap="viridis")
        bar.set_array([0, 1])
        return bar

    avg_ratings, num_reviews = np.meshgrid(np.linspace(1, 5, num=res), np.linspace(0, 200, num=res))
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey='all')
    for idx, rating in enumerate(['D', 'DD', 'DDD', 'DDDD']):
        ctr = ctr_estimate(avg_ratings.flatten(), num_reviews.flatten(), [rating] * (res ** 2))
        axes[idx].pcolor(avg_ratings, num_reviews, ctr.reshape(res, res), shading='auto', vmin=0, vmax=1)
        axes[idx].set(title=rating, xlabel='Average Rating', ylabel=None if idx > 0 else 'Number of Reviews')
    fig.colorbar(color_bar(), cax=fig.add_axes([0.92, 0.15, 0.01, 0.7]))
    fig.suptitle(title)


def plot_histograms(figsize=(10, 8), tight_layout=True, **kwargs):
    if len(kwargs) > 0:
        fig, axes = plt.subplots(len(kwargs), 3, sharex='col', sharey='row', figsize=figsize, tight_layout=tight_layout)
        axes = axes.reshape((-1, 3))
        for i, (title, x) in enumerate(kwargs.items()):
            sns.histplot(x=x['avg_rating'], ax=axes[i, 0]).set(ylabel=title.capitalize())
            sns.histplot(x=x['num_reviews'], ax=axes[i, 1])
            sns.histplot(x=x[['D', 'DD', 'DDD', 'DDDD']].idxmax(axis=1), ax=axes[i, 2])


def plot_distributions(figsize=(10, 8), tight_layout=True, **kwargs):
    if len(kwargs) > 0:
        fig, axes = plt.subplots(len(kwargs), 3, sharex='col', figsize=figsize, tight_layout=tight_layout)
        axes = axes.reshape((-1, 3))
        for i, (title, (x, y)) in enumerate(kwargs.items()):
            sns.kdeplot(x=x['avg_rating'], hue=y, ax=axes[i, 0]).set(ylabel=title.capitalize())
            sns.kdeplot(x=x['num_reviews'], hue=y, ax=axes[i, 1])
            sns.countplot(x=x[['D', 'DD', 'DDD', 'DDDD']].idxmax(axis=1), hue=y, ax=axes[i, 2])


def plot_conclusions(models, figsize=(10, 10), res=100):
    avg_ratings, num_reviews = np.meshgrid(np.linspace(1, 5, num=res), np.linspace(0, 200, num=res))
    fig, axes = plt.subplots(4, len(models), figsize=figsize, sharex='all', sharey='all', tight_layout=True)
    axes = axes.reshape((4, -1))
    for i, rating in enumerate(['D', 'DD', 'DDD', 'DDDD']):
        for j, (title, model) in enumerate(models.items()):
            ctr = model.ctr_estimate(avg_ratings.flatten(), num_reviews.flatten(), [rating] * (res ** 2))
            axes[i, j].pcolor(avg_ratings, num_reviews, ctr.reshape(100, 100), shading='auto', vmin=0, vmax=1)
            axes[i, j].set(title=title if i == 0 else None, xlabel=None, ylabel=rating if j == 0 else None)
