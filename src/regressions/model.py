import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from src.models import Model
from src.regressions.data import synthetic_function
from src.util.preprocessing import Scaler


def import_extension_methods():
    def synthetic_summary(model, scalers=None, res=50, figsize=(14, 8), tight_layout=True, **kwargs):
        x_scaler, y_scaler = (Scaler.get_default(2), Scaler.get_default(1)) if scalers is None else scalers
        # compute metrics on kwargs
        summary = []
        for title, (x, y) in kwargs.items():
            p = y_scaler.invert(model.predict(x))
            y = y_scaler.invert(y)
            summary.append(f'{r2_score(y, p):.4} ({title} r2)')
        print(', '.join(summary))
        # estimated functions
        a, b = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
        grid = pd.DataFrame.from_dict({'a': a.flatten(), 'b': b.flatten()})
        grid['pred'] = y_scaler.invert(model.predict(x_scaler.transform(grid)))
        grid['label'] = synthetic_function(grid['a'], grid['b'])
        _, axes = plt.subplots(2, 3, figsize=figsize, tight_layout=tight_layout)
        for ax, (title, y) in zip(axes, {'Ground Truth': 'label', 'Estimated Function': 'pred'}.items()):
            # plot bivariate function
            z = grid[y].values.reshape(res, res)
            ax[0].pcolor(a, b, z, shading='auto', cmap='viridis', vmin=grid['label'].min(), vmax=grid['label'].max())
            # plot first feature (with title as it is the central plot)
            for _, group in grid.groupby('b'):
                sns.lineplot(data=group, x='a', y=y, color='#CCC', alpha=0.4, ax=ax[1])
            sns.lineplot(data=grid, x='a', y=y, color='black', ci=None, label='average', ax=ax[1]).set(title=title)
            # plot second feature
            for _, group in grid.groupby('a'):
                sns.lineplot(data=group, x='b', y=y, color='#CCC', alpha=0.4, ax=ax[2])
            sns.lineplot(data=grid, x='b', y=y, color='black', ci=None, label='average', ax=ax[2])

    def cars_summary(model, scalers=None, res=100, xlim=(0, 60), ylim=(0, 120), figsize=(10, 4), **kwargs):
        plt.figure(figsize=figsize)
        x_scaler, y_scaler = (Scaler.get_default(1), Scaler.get_default(1)) if scalers is None else scalers
        # evaluation data
        summary = []
        for title, (x, y) in kwargs.items():
            p = y_scaler.invert(model.predict(x))
            x = x_scaler.invert(x['price'])
            y = y_scaler.invert(y)
            summary.append(f'{r2_score(y, p):.4} ({title} r2)')
            sns.scatterplot(x=x, y=y, alpha=0.25, sizes=0.25, label=title.capitalize())
        print(', '.join(summary))
        # estimated function
        x = np.linspace(x_scaler.transform(xlim)[0], x_scaler.transform(xlim)[1], res)
        y = model.predict(x).flatten()
        sns.lineplot(x=x_scaler.invert(x), y=y_scaler.invert(y), color='black').set(
            xlabel='price', ylabel='sales', title='Estimated Function'
        )
        plt.xlim(xlim)
        plt.ylim(ylim)

    def puzzles_summary(model, scalers=None, res=5, figsize=(14, 4), tight_layout=True, **kwargs):
        features = ['word_count', 'star_rating', 'num_reviews']
        fig, axes = plt.subplots(1, 3, sharey='all', tight_layout=tight_layout, figsize=figsize)
        x_scaler, y_scaler = (Scaler.get_default(3), Scaler.get_default(1)) if scalers is None else scalers
        # evaluation data
        summary = []
        for title, (x, y) in kwargs.items():
            p = y_scaler.invert(model.predict(x))
            y = y_scaler.invert(y)
            summary.append(f'{r2_score(y, p):.4} ({title} r2)')
        print(', '.join(summary))
        # estimated functions
        grid = np.meshgrid(np.linspace(0, 1, res), np.linspace(0, 1, res), np.linspace(0, 1, res))
        grid = np.concatenate([x.reshape(-1, 1) for x in grid], axis=1)
        grid = pd.DataFrame(grid, columns=features)
        pred = y_scaler.invert(model.predict(grid)).flatten()
        grid = x_scaler.invert(grid)
        grid['pred'] = pred
        for ax, feat in zip(axes, features):
            # plot predictions for each group of other features
            for _, group in grid.groupby([c for c in grid.columns if c not in [feat, 'pred']]):
                sns.lineplot(data=group, x=feat, y='pred', color='#CCC', ax=ax)
            # plot mean prediction
            sns.lineplot(data=grid, x=feat, y='pred', color='black', ci=None, label='average', ax=ax).set(
                xlim=(grid[feat].min(), grid[feat].max()), ylim=(pred.min(), pred.max())
            )
        fig.suptitle('Estimated Functions')

    Model.synthetic_summary = synthetic_summary
    Model.cars_summary = cars_summary
    Model.puzzles_summary = puzzles_summary
