"""Script to get Plots from Preliminary Tests."""
import matplotlib.pyplot as plt
import numpy as np
import wandb
import pandas as pd
import seaborn as sns

LINE_WIDTH = 5
FIG_SIZE = (25, 10)
FONT_SCALE = 2.0

COLUMNS = {
    'name': 'dataset',
    'iteration': 'iteration',
    'fold': 'fold',
    'use_prob': 'Use Probabilities',
    'init': 'Initial Step',
    'learner-train/Accuracy': 'Accuracy',
    'learner-train/Std. Dev. of class frequencies': 'Std. Dev. of Class Frequencies',
    'learner-train/R2': 'R2',
    'learner-train/DIDI perc. index': 'DIDI Percentage Index'
}


def _plot_config(fig, title):
    fig.set_xticks([t for t in np.unique(fig.get_xticks().round().astype(int)) if t in range(16)])
    fig.set_ylabel('')
    fig.set_title(title)


if __name__ == '__main__':
    sns.set_style('whitegrid')
    sns.set_context('talk', font_scale=FONT_SCALE)
    plt.rc('lines', linewidth=LINE_WIDTH)

    df = []
    runs = wandb.Api().runs('giuluck/mt_preliminary')
    for run in runs:
        history = run.history().fillna(method='ffill')
        history['name'] = run.name
        df.append(history)
    df = pd.concat(df)[list(COLUMNS.keys())].rename(columns=COLUMNS).reset_index(drop=True)

    _, axes = plt.subplots(1, 2, figsize=FIG_SIZE, tight_layout=True)
    data = df[df['dataset'] == 'redwine']
    for ax, y in zip(axes, ['Accuracy', 'Std. Dev. of Class Frequencies']):
        sns.lineplot(data=data, x='iteration', y=y, style='Use Probabilities', hue='Initial Step', ci='sd', ax=ax)
        _plot_config(ax, y)
    plt.savefig('../temp/redwine.png', format='png')

    _, axes = plt.subplots(1, 2, figsize=FIG_SIZE, tight_layout=True)
    data = df[df['dataset'] == 'adult']
    for ax, y in zip(axes, ['Accuracy', 'DIDI Percentage Index']):
        sns.lineplot(data=data, x='iteration', y=y, style='Use Probabilities', hue='Initial Step', ci='sd', ax=ax)
        _plot_config(ax, y)
    plt.savefig('../temp/adult.png', format='png')

    _, axes = plt.subplots(1, 2, figsize=FIG_SIZE, tight_layout=True)
    data = df[df['dataset'] == 'crime']
    for ax, y in zip(axes, ['R2', 'DIDI Percentage Index']):
        sns.lineplot(data=data, x='iteration', y=y, hue='Initial Step', ci='sd', ax=ax)
        _plot_config(ax, y)
    plt.savefig('../temp/crime.png', format='png')
