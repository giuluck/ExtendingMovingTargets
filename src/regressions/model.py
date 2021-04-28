import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from moving_targets.metrics.constraints import MonotonicViolation
from src.models import Model
from src.regressions.augmentation import compute_monotonicities
from src.regressions.data import synthetic_function
from src.util.augmentation import get_monotonicities_list
from src.util.plot import ColorFader
from src.util.preprocessing import Scaler


def metrics_summary(model, y_scaler, **kwargs):
    summary = []
    for title, (x, y) in kwargs.items():
        p = y_scaler.invert(model.predict(x))
        y = y_scaler.invert(y)
        summary.append(f'{r2_score(y, p):.4} ({title} r2)')
    return ', '.join(summary)


def violations_summary(model, grid, monotonicities, y_scaler):
    p = y_scaler.invert(model.predict(grid))
    avg_violation = MonotonicViolation(monotonicities=monotonicities, aggregation='average', eps=0.0)
    pct_violation = MonotonicViolation(monotonicities=monotonicities, aggregation='percentage', eps=0.0)
    return f'{avg_violation(None, None, p):.4} (avg. violation), {pct_violation(None, None, p):.4} (pct. violation)'


def synthetic_summary(model, scalers=None, res=50, figsize=(14, 8), tight_layout=True, **kwargs):
    x_scaler, y_scaler = (Scaler.get_default(2), Scaler.get_default(1)) if scalers is None else scalers
    # compute metrics on kwargs
    violations_grid, mono = model.grids['synthetic']
    print(violations_summary(model, grid=violations_grid, monotonicities=mono, y_scaler=y_scaler))
    print(metrics_summary(model, y_scaler, **kwargs))
    # estimated functions
    a, b = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
    grid = pd.DataFrame.from_dict({'a': a.flatten(), 'b': b.flatten()})
    grid['pred'] = y_scaler.invert(model.predict(x_scaler.transform(grid)))
    grid['label'] = synthetic_function(grid['a'], grid['b'])
    fader = ColorFader('red', 'blue', bounds=(-1, 1))
    _, axes = plt.subplots(2, 3, figsize=figsize, tight_layout=tight_layout)
    for ax, (title, y) in zip(axes, {'Ground Truth': 'label', 'Estimated Function': 'pred'}.items()):
        # plot bivariate function
        z = grid[y].values.reshape(res, res)
        ax[0].pcolor(a, b, z, shading='auto', cmap='viridis', vmin=grid['label'].min(), vmax=grid['label'].max())
        # plot first feature (with title as it is the central plot)
        for idx, group in grid.groupby('b'):
            label = f'b = {idx:.0f}' if idx in [-1, 1] else None
            sns.lineplot(data=group, x='a', y=y, color=fader(idx), alpha=0.4, label=label, ax=ax[1])
        # plot second feature
        for idx, group in grid.groupby('a'):
            label = f'a = {idx:.0f}' if idx in [-1, 1] else None
            sns.lineplot(data=group, x='b', y=y, color=fader(idx), alpha=0.4, label=label, ax=ax[2])
    plt.show()


def cars_summary(model, scalers=None, res=100, xlim=(0, 60), ylim=(0, 120), figsize=(10, 4), **kwargs):
    plt.figure(figsize=figsize)
    x_scaler, y_scaler = (Scaler.get_default(1), Scaler.get_default(1)) if scalers is None else scalers
    # compute metrics on kwargs and plot data points
    violations_grid, mono = model.grids['cars']
    print(violations_summary(model, grid=violations_grid, monotonicities=mono, y_scaler=y_scaler))
    print(metrics_summary(model, y_scaler, **kwargs))
    for title, (x, y) in kwargs.items():
        x, y = x_scaler.invert(x['price']), y_scaler.invert(y)
        sns.scatterplot(x=x, y=y, alpha=0.25, sizes=0.25, label=title.capitalize())
    # estimated function
    x = np.linspace(x_scaler.transform(xlim)[0], x_scaler.transform(xlim)[1], res)
    y = model.predict(x.reshape(-1, 1)).flatten()
    sns.lineplot(x=x_scaler.invert(x), y=y_scaler.invert(y), color='black').set(
        xlabel='price', ylabel='sales', title='Estimated Function'
    )
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()


def puzzles_summary(model, scalers=None, res=5, figsize=(14, 4), tight_layout=True, **kwargs):
    features = ['word_count', 'star_rating', 'num_reviews']
    fig, axes = plt.subplots(1, 3, sharey='all', tight_layout=tight_layout, figsize=figsize)
    x_scaler, y_scaler = (Scaler.get_default(3), Scaler.get_default(1)) if scalers is None else scalers
    # compute metrics on kwargs
    violations_grid, mono = model.grids['puzzles']
    print(violations_summary(model, grid=violations_grid, monotonicities=mono, y_scaler=y_scaler))
    print(metrics_summary(model, y_scaler, **kwargs))
    # estimated functions
    grid = np.meshgrid(np.linspace(0, 1, res), np.linspace(0, 1, res), np.linspace(0, 1, res))
    grid = pd.DataFrame.from_dict({k: v.flatten() for k, v in zip(features, grid)})
    pred = y_scaler.invert(model.predict(grid)).flatten()
    grid = x_scaler.invert(grid)
    grid['pred'] = pred
    for ax, feat in zip(axes, features):
        # plot predictions for each group of other features
        fi, fj = [f for f in grid.columns if f not in [feat, 'pred']]
        li, ui = grid[fi].min(), grid[fi].max()
        lj, uj = grid[fj].min(), grid[fj].max()
        fader = ColorFader('black', 'magenta', 'cyan', 'yellow', bounds=(li, lj, ui, uj))
        for (i, j), group in grid.groupby([fi, fj]):
            label = f'{fi}: {i:.0f}, {fj}: {j:.0f}' if (i in [li, ui] and j in [lj, uj]) else None
            sns.lineplot(data=group, x=feat, y='pred', color=fader(i, j), alpha=0.6, label=label, ax=ax)
    fig.suptitle('Estimated Functions')
    plt.show()


def import_extension_methods(synthetic_res=80, cars_res=700, puzzles_res=20):
    grids = {
        'synthetic': (synthetic_res, [1, 0]),
        'cars': (cars_res, [-1]),
        'puzzles': (puzzles_res, [-1, 1, 1])
    }
    Model.grids = {}
    for key, (res, directions) in grids.items():
        grid = np.meshgrid(*([np.linspace(0, 1, res)] * len(directions)))
        grid = pd.DataFrame.from_dict({str(k): v.flatten() for k, v in enumerate(grid)})
        Model.grids[key] = (
            grid,
            get_monotonicities_list(
                data=grid, label=None, kinds='all', errors='ignore',
                compute_monotonicities=lambda s, r: compute_monotonicities(s, r, directions=directions)
            )
        )
    Model.metrics_summary = metrics_summary
    Model.violations_summary = violations_summary
    Model.synthetic_summary = synthetic_summary
    Model.cars_summary = cars_summary
    Model.puzzles_summary = puzzles_summary


if __name__ == '__main__':
    from src.models import MLP
    from src.regressions.data import load_cars

    import_extension_methods(synthetic_res=5, cars_res=5, puzzles_res=5)
    data = load_cars('../../res/cars.csv')
    x_train, y_train = data['train']

    mlp = MLP(output_act=None, h_units=[16] * 4)
    mlp.compile(optimizer='adam', loss='mse')
    mlp.fit(x_train, y_train, epochs=500)
    mlp.cars_summary(**data)
