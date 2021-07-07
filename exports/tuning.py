"""Script to get Plots from Tuning Tests."""
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.datasets import SyntheticManager, RestaurantsManager
from src.util.plot import ColorFader

LINE_WIDTH = 5
ALPHA = 0.7
FIG_SIZES = [(25, 8), (25, 15), (34, 17)]
FONT_SCALE = 2.0

COLUMNS = {
    'name': 'experiment',
    'iteration': 'iteration',
    'fold': 'fold',
    'init_step': 'Initial Step',
    'mono/kind': 'Monotonicities',
    'master/alpha': 'Alpha',
    'master/beta': 'Beta',
    'master/learner_omega': 'Learner Omega',
    'master/master_omega': 'Master Omega',
    'master/learner_weights': 'Learner Weights',
    'master/loss_fn': 'Master Loss',
    'learner/loss': 'Learner Loss',
    'learner/warm_start': 'Warm Start',
    'metrics/train metric': 'Train Metric',
    'metrics/validation metric': 'Validation Metric',
    'metrics/test metric': 'Test Metric',
    'metrics/avg. violation': 'Average Violation',
    'time/iteration': 'Training Time'
}


def _plot_config(figure, title):
    figure.set_xticks([t for t in np.unique(figure.get_xticks().round().astype(int)) if t in range(31)])
    figure.set_ylabel('')
    figure.set_title(title)


def _plot_legend(axis, **kwargs):
    sns.lineplot(ax=axis, **kwargs).set(xlabel='', xticks=[], yticks=[])
    plt.legend(loc='center')
    for spine in axis.spines.values():
        spine.set_visible(False)


if __name__ == '__main__':
    sns.set_context('talk', font_scale=FONT_SCALE)
    plt.rc('lines', linewidth=LINE_WIDTH)

    a, b = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    z = SyntheticManager.function(a.flatten(), b.flatten())
    grid = pd.DataFrame.from_dict({'a': a.flatten(), 'b': b.flatten(), 'label': z})
    fader = ColorFader('red', 'blue', bounds=[-1, 1])
    fig, ax = plt.subplots(1, 3, figsize=FIG_SIZES[0], tight_layout=True)
    ax[0].pcolor(a, b, z.reshape(100, 100), shading='auto', cmap='viridis', vmin=z.min(), vmax=z.max())
    ax[0].set_xlabel('a')
    ax[0].set_ylabel('b')
    for idx, group in grid.groupby('b'):
        label = f'b = {idx:.0f}' if idx in [-1, 1] else None
        sns.lineplot(data=group, x='a', y='label', color=fader(idx), alpha=0.4, label=label, ax=ax[1])
    for idx, group in grid.groupby('a'):
        label = f'a = {idx:.0f}' if idx in [-1, 1] else None
        sns.lineplot(data=group, x='b', y='label', color=fader(idx), alpha=0.4, label=label, ax=ax[2])
    plt.savefig('../temp/synthetic.png', format='png')

    avg_ratings, num_reviews = np.meshgrid(np.linspace(1, 5, num=100), np.linspace(0, 200, num=100))
    _, axes = plt.subplots(1, 4, sharex='all', sharey='all', figsize=FIG_SIZES[0], tight_layout=True)
    for i, rating in enumerate(['D', 'DD', 'DDD', 'DDDD']):
        df = pd.DataFrame({'avg_rating': avg_ratings.flatten(), 'num_reviews': num_reviews.flatten()})
        for d in ['D', 'DD', 'DDD', 'DDDD']:
            df[d] = 1 if rating == d else 0
        ctr = RestaurantsManager.predict(df).reshape(100, 100)
        axes[i].pcolor(avg_ratings, num_reviews, ctr, shading='auto', cmap='viridis', vmin=0, vmax=1)
        axes[i].set(title=rating, xlabel='avg_rating', ylabel='num_reviews' if i == 0 else None)
    plt.savefig('../temp/restaurants.png', format='png')

    df = []
    sns.set_style('whitegrid')
    runs = wandb.Api().runs('giuluck/mt_tuning')
    for run in runs:
        history = pd.DataFrame.from_dict({k: v for k, v in run.config.items() if k in COLUMNS.keys()}, orient='index')
        history = history.transpose().fillna('None').astype('string')
        history = pd.concat((history, run.history().drop(columns='learner/loss')), axis=1).fillna(method='ffill')
        history['name'] = run.name
        df.append(history)
    df = pd.concat(df)[list(COLUMNS.keys())].rename(columns=COLUMNS).reset_index(drop=True)
    df.to_csv('../temp/tuning.csv')

    excluded = df[df['experiment'] == 'preliminary']
    excluded = excluded[excluded['fold'] == 0]
    excluded = excluded[excluded['Monotonicities'] != 'ground']
    excluded = excluded[excluded['Initial Step'] == 'projection']
    excluded = excluded[excluded['Alpha'] == '0.1']
    excluded = excluded[excluded['Beta'] == '1']
    excluded = excluded[excluded['Master Loss'] == 'mae']
    df = df.drop(index=excluded.index)

    data = {'preliminary': ['Initial Step', 'Monotonicities'], 'regression': ['Learner Omega', 'Master Loss']}
    for experiment, hues in data.items():
        data = df[df['experiment'] == experiment].rename(columns=lambda c: c.replace('Metric', 'R2'))
        for hue in hues:
            _, axes = plt.subplots(2, 2, figsize=FIG_SIZES[1], tight_layout=True)
            for ax, y in zip(axes.flatten(), ['Train R2', 'Validation R2', 'Test R2', 'Average Violation']):
                sns.lineplot(data=data, x='iteration', y=y, hue=hue, ci='sd', alpha=ALPHA, ax=ax)
                _plot_config(ax, y)
            plt.savefig(f"../temp/{hue.lower().replace(' ', '_')}.png", format='png')

    data = df[df['experiment'] == 'classification'].rename(columns=lambda c: c.replace('Metric', 'AUC'))
    axes = plt.subplots(2, 3, figsize=FIG_SIZES[2], tight_layout=True)[1].flatten()
    for ax, y in zip(axes, ['Test AUC', 'Validation AUC', 'Train AUC', 'Average Violation', 'Training Time']):
        sns.lineplot(data=data, x='iteration', y=y, hue='Master Loss', style='Learner Loss',
                     style_order=['binary_crossentropy', 'mse'], ci=None, legend=None, ax=ax)
        _plot_config(ax, y)
    _plot_legend(axis=axes[-1], data=data, x='iteration', y=np.nan, hue='Master Loss',
                 style='Learner Loss', style_order=['binary_crossentropy', 'mse'])
    plt.savefig(f"../temp/classification.png", format='png')
