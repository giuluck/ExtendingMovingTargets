"""Moving Targets Callbacks."""

from typing import List, Any, Dict, Union, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame

from moving_targets.callbacks import Callback
from moving_targets.util.typing import Matrix, Vector, Dataset, Iteration
from src.datasets import RestaurantsManager, SyntheticManager
from src.util.plot import ColorFader

YInfo = Union[Vector, DataFrame]


# noinspection PyMissingOrEmptyDocstring
class AnalysisCallback(Callback):
    PRETRAINING = 'PT'
    MARKERS = dict(aug='o', label='X')

    def __init__(self, num_columns: int = 5, sorting_attribute: object = None, file_signature: str = None,
                 do_plot: bool = True, **kwargs):
        super(AnalysisCallback, self).__init__()
        self.num_columns: int = num_columns
        self.sorting_attribute: object = sorting_attribute
        self.file_signature: str = file_signature
        self.do_plot: bool = do_plot
        self.plt_kwargs: Dict = {'figsize': (20, 10), 'tight_layout': True}
        self.plt_kwargs.update(kwargs)
        self.data: Optional[pd.DataFrame] = None
        self.iterations: List[Any] = []

    def on_process_start(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], **additional_kwargs):
        m = pd.Series(['aug' if m else 'label' for m in macs.master.augmented_mask], name='mask')
        self.data = pd.concat((x.reset_index(drop=True), y.reset_index(drop=True), m), axis=1)

    def on_pretraining_start(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], **additional_kwargs):
        additional_kwargs['iteration'] = AnalysisCallback.PRETRAINING
        self.on_iteration_start(macs, x, y, val_data, **additional_kwargs)
        self.on_adjustment_start(macs, x, y, val_data, **additional_kwargs)
        self.on_adjustment_end(macs, x, y, np.ones_like(y) * np.nan, val_data, **additional_kwargs)
        self.on_training_start(macs, x, y, val_data, **additional_kwargs)

    def on_pretraining_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], **additional_kwargs):
        additional_kwargs['iteration'] = AnalysisCallback.PRETRAINING
        self.on_training_end(macs, x, y, val_data, **additional_kwargs)
        self.on_iteration_end(macs, x, y, val_data, **additional_kwargs)

    def on_iteration_start(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration,
                           **additional_kwargs):
        self.iterations.append(iteration)
        self.data[f'y {iteration}'] = y

    def on_process_end(self, macs, val_data: Optional[Dataset], **additional_kwargs):
        # sort values
        if self.sorting_attribute is not None:
            self.data = self.data.sort_values(self.sorting_attribute)
        # write on files
        if self.file_signature is not None:
            self.data.to_csv(self.file_signature + '.csv', index_label='index')
            with open(self.file_signature + '.txt', 'w') as f:
                f.write(str(self.data))
        # do plots
        if self.do_plot:
            plt.figure(**self.plt_kwargs)
            num_rows = int(np.ceil(len(self.iterations) / self.num_columns))
            ax = None
            for idx, it in enumerate(self.iterations):
                ax = plt.subplot(num_rows, self.num_columns, idx + 1, sharex=ax, sharey=ax)
                title = self.plot_function(it)
                ax.set(xlabel='', ylabel='')
                ax.set_title(f'{it})' if title is None else title)
            plt.show()

    def plot_function(self, iteration: Iteration) -> Optional[str]:
        pass


# noinspection PyMissingOrEmptyDocstring
class DistanceAnalysis(AnalysisCallback):
    def __init__(self, ground_only: bool = True, num_columns=1, **kwargs):
        super(DistanceAnalysis, self).__init__(num_columns=num_columns, **kwargs)
        self.ground_only: bool = ground_only
        self.y: Optional[str] = None

    def on_pretraining_start(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], **additional_kwargs):
        self.y = y.name
        super(DistanceAnalysis, self).on_pretraining_start(macs, x, y, val_data, **additional_kwargs)

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration,
                        **additional_kwargs):
        self.data[f'pred {iteration}'] = macs.predict(x)

    def on_adjustment_end(self, macs, x: Matrix, y: Vector, adjusted_y: Vector, val_data: Optional[Dataset],
                          iteration: Iteration, **additional_kwargs):
        self.data[f'adj {iteration}'] = adjusted_y

    def on_process_end(self, macs, val_data: Optional[Dataset], **additional_kwargs):
        if self.ground_only:
            self.data = self.data[self.data['mask'] == 'label']
        super(DistanceAnalysis, self).on_process_end(macs, val_data)

    def plot_function(self, iteration: Iteration) -> Optional[str]:
        x = np.arange(len(self.data))
        y, p, j = self.data[self.y].values, self.data[f'pred {iteration}'].values, self.data[f'adj {iteration}'].values
        style = self.data['mask']
        sns.scatterplot(x=x, y=y, color='black', alpha=0.6).set_xticks([])
        sns.scatterplot(x=x, y=p, color='red', alpha=0.6)
        sns.scatterplot(x=x, y=j, style=style, markers=AnalysisCallback.MARKERS, color='blue', alpha=0.8, s=50)
        plt.legend(['labels', 'predictions', 'adjusted'])
        for i in x:
            plt.plot([i, i], [p[i], j[i]], c='red')
            plt.plot([i, i], [y[i], j[i]], c='black')
        avg_pred_distance = np.abs(p - j).mean()
        avg_label_distance = np.abs(y[style == 'label'] - j[style == 'label']).mean()
        return f'{iteration}) pred. distance = {avg_pred_distance:.4f}, label distance = {avg_label_distance:.4f}'


# noinspection PyMissingOrEmptyDocstring
class BoundsAnalysis(AnalysisCallback):
    def __init__(self, num_columns: int = 1, **kwargs):
        super(BoundsAnalysis, self).__init__(num_columns=num_columns, **kwargs)

    def on_process_start(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], **additional_kwargs):
        super(BoundsAnalysis, self).on_process_start(macs, x, y, val_data, **additional_kwargs)
        hi, li = macs.master.higher_indices, macs.master.lower_indices
        self.data['lower'] = self.data.index.map(lambda i: li[hi == i])
        self.data['higher'] = self.data.index.map(lambda i: hi[li == i])

    def on_pretraining_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], **additional_kwargs):
        pass

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration,
                        **additional_kwargs):
        self._insert_bounds(macs.predict(x), 'pred', iteration)

    def on_adjustment_end(self, macs, x: Matrix, y: Vector, adjusted_y: Vector, val_data: Optional[Dataset],
                          iteration: Iteration, **additional_kwargs):
        self._insert_bounds(adjusted_y, 'adj', iteration)

    def _insert_bounds(self, v: np.ndarray, label: str, iteration: Iteration):
        self.data[f'{label} {iteration}'] = v
        self.data[f'{label} lb {iteration}'] = self.data['lower'].map(lambda i: v[i].max() if len(i) > 0 else None)
        self.data[f'{label} ub {iteration}'] = self.data['higher'].map(lambda i: v[i].min() if len(i) > 0 else None)

    def plot_function(self, iteration: Iteration) -> Optional[str]:
        x = np.arange(len(self.data))
        avg_bound = {}
        for label, color in dict(adj='blue', pred='red').items():
            val = self.data[f'{label} {iteration}']
            lb = self.data[f'{label} lb {iteration}'].fillna(val.min())
            ub = self.data[f'{label} ub {iteration}'].fillna(val.max())
            sns.scatterplot(x=x, y=lb, marker='^', color=color, alpha=0.4)
            sns.scatterplot(x=x, y=ub, marker='v', color=color, alpha=0.4)
            sns.scatterplot(x=x, y=val, color=color, edgecolors='black', label=label).set_xticks([])
            avg_bound[label] = np.mean(ub - lb)
        return f'{iteration}) ' + ', '.join([f'{k} bound = {v:.2f}' for k, v in avg_bound.items()])


# noinspection PyMissingOrEmptyDocstring
class CarsAdjustments(AnalysisCallback):
    max_size = 50
    alpha = 0.4

    def __init__(self, plot_kind: str = 'scatter', **kwargs):
        super(CarsAdjustments, self).__init__(**kwargs)
        assert plot_kind in ['line', 'scatter'], f"'{plot_kind}' is not a valid plot kind"
        self.plot_kind: str = plot_kind

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration,
                        **additional_kwargs):
        self.data[f'pred {iteration}'] = macs.predict(x)

    def on_adjustment_end(self, macs, x: Matrix, y: Vector, adjusted_y: Vector, val_data: Optional[Dataset],
                          iteration: Iteration, **additional_kwargs):
        self.data[f'adj {iteration}'] = adjusted_y
        self.data[f'sw {iteration}'] = additional_kwargs.get('sample_weight',
                                                             np.where(self.data['mask'] == 'label', 1, 0))

    def plot_function(self, iteration: Iteration) -> Optional[str]:
        x, y = self.data['price'].values, self.data['sales'].values
        s, m = np.array(self.data['mask']), dict(aug='o', label='X')
        sn, al = (0, CarsAdjustments.max_size), CarsAdjustments.alpha
        p, adj, sw = self.data[f'pred {iteration}'], self.data[f'adj {iteration}'], self.data[f'sw {iteration}'].values
        # rescale original labels weights
        if iteration == AnalysisCallback.PRETRAINING:
            sns.scatterplot(x=x, y=y, style=s, markers=m, size=sw, size_norm=(0, 1), sizes=sn, color='black', alpha=al)
        elif self.plot_kind == 'line':
            sns.lineplot(x=x, y=adj, color='blue')
            for i in range(self.data.shape[0]):
                plt.plot([x[i], x[i]], [y[i], adj[i]], c='black', alpha=al)
        elif self.plot_kind == 'scatter':
            sns.scatterplot(x=x, y=adj, style=s, markers=m, size=sw, size_norm=(0, 1), sizes=sn, color='blue', alpha=al)
        sns.lineplot(x=x, y=p, color='red')
        plt.legend(['predictions', 'labels' if iteration == AnalysisCallback.PRETRAINING else 'adjusted'])
        return f'{iteration}) adj. mse = {np.square((adj - y).fillna(0)).mean():.4f}'


# noinspection PyMissingOrEmptyDocstring
class DefaultAdjustments(AnalysisCallback):
    max_size = 30
    alpha = 0.8

    def __init__(self, **kwargs):
        super(DefaultAdjustments, self).__init__(**kwargs)
        married, payment = np.meshgrid([0, 1], np.arange(-2, 9))
        self.grid = pd.DataFrame.from_dict({'married': married.flatten(), 'payment': payment.flatten()})

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration,
                        **additional_kwargs):
        self.data[f'pred {iteration}'] = macs.predict(x)
        self.grid[f'pred {iteration}'] = macs.predict(self.grid[['married', 'payment']])

    def on_adjustment_end(self, macs, x: Matrix, y: Vector, adjusted_y: Vector, val_data: Optional[Dataset],
                          iteration: Iteration, **additional_kwargs):
        self.data[f'adj {iteration}'] = adjusted_y
        self.data[f'sw {iteration}'] = additional_kwargs.get('sample_weight',
                                                             np.where(self.data['mask'] == 'label', 1, 0))

    def plot_function(self, iteration: Iteration) -> Optional[str]:
        y, adj = self.data['default'], self.data[f'adj {iteration}']
        label = 'default' if iteration == AnalysisCallback.PRETRAINING else f'adj {iteration}'
        data = self.data.astype({'married': int}).rename(columns={label: 'y', f'sw {iteration}': 'sw'})
        data = data.groupby(['married', 'payment', 'y'])['sw'].sum().reset_index()
        # dodge based on married value due to visualization reasons
        data['payment'] = [d + (0.1 if m == 0 else -0.1) for d, m in zip(data['payment'], data['married'])]
        # plot mass probabilities and responses
        sns.scatterplot(data=data, x='payment', y='y', hue='married', size='sw', size_norm=(0, 100), legend=False,
                        sizes=(0, DefaultAdjustments.max_size), alpha=DefaultAdjustments.alpha)
        sns.lineplot(data=self.grid, x='payment', y=f'pred {iteration}', hue='married')
        return f'{iteration}) avg. flips = {np.abs(adj[~np.isnan(y)] - y[~np.isnan(y)]).mean():.4f}'


# noinspection PyMissingOrEmptyDocstring
class LawAdjustments(AnalysisCallback):
    max_size = 40

    def __init__(self, res: int = 100, data_points: bool = True, **kwargs):
        super(LawAdjustments, self).__init__(**kwargs)
        lsat, ugpa = np.meshgrid(np.linspace(0, 50, res), np.linspace(0, 4, res))
        self.grid: pd.DataFrame = pd.DataFrame.from_dict({'lsat': lsat.flatten(), 'ugpa': ugpa.flatten()})
        self.res: int = res
        self.data_points: bool = data_points

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration,
                        **additional_kwargs):
        grid = self.grid[['lsat', 'ugpa']]
        data = self.data[['lsat', 'ugpa']]
        self.grid[f'pred {iteration}'] = macs.predict(grid)
        self.data[f'pred {iteration}'] = macs.predict(data)

    def on_adjustment_end(self, macs, x: Matrix, y: Vector, adjusted_y: Vector, val_data: Optional[Dataset],
                          iteration: Iteration, **additional_kwargs):
        self.data[f'adj {iteration}'] = adjusted_y
        self.data[f'sw {iteration}'] = additional_kwargs.get('sample_weight',
                                                             np.where(self.data['mask'] == 'label', 1, 0))

    def plot_function(self, iteration: Iteration) -> Optional[str]:
        # plot 3D response
        lsat = self.grid['lsat'].values.reshape(self.res, self.res)
        ugpa = self.grid['ugpa'].values.reshape(self.res, self.res)
        pred = self.grid[f'pred {iteration}'].values.reshape(self.res, self.res)
        plt.pcolor(lsat, ugpa, pred, shading='auto', vmin=0, vmax=1)
        # plot data points
        if self.data_points:
            markers, sizes = AnalysisCallback.MARKERS, (0, LawAdjustments.max_size)
            sns.scatterplot(data=self.data, x='lsat', y='ugpa', size=f'sw {iteration}', size_norm=(0, 1),
                            sizes=sizes, color='black', style='mask', markers=markers, legend=False)
        return


# noinspection PyMissingOrEmptyDocstring
class LawResponse(AnalysisCallback):
    def __init__(self, feature: str, res: int = 100, **kwargs):
        super(LawResponse, self).__init__(**kwargs)
        assert feature in ['lsat', 'ugpa'], f"'{feature}' is not a valid feature"
        lsat, ugpa = np.meshgrid(np.linspace(0, 50, res), np.linspace(0, 4, res))
        self.grid: pd.DataFrame = pd.DataFrame.from_dict({'lsat': lsat.flatten(), 'ugpa': ugpa.flatten()})
        self.fader: ColorFader = ColorFader('red', 'blue', bounds=[0, 4] if feature == 'lsat' else [0, 50])
        self.features: Tuple[str, str] = ('lsat', 'ugpa') if feature == 'lsat' else ('ugpa', 'lsat')

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration,
                        **additional_kwargs):
        input_grid = self.grid[['lsat', 'ugpa']]
        self.grid[f'pred {iteration}'] = macs.predict(input_grid)

    def plot_function(self, iteration: Iteration) -> Optional[str]:
        feat, group_feat = self.features
        for group_val, group in self.grid.groupby([group_feat]):
            sns.lineplot(data=group, x=feat, y=f'pred {iteration}', color=self.fader(group_val), alpha=0.6)
        return f'{iteration}) {feat.upper()}'


# noinspection PyMissingOrEmptyDocstring
class PuzzlesResponse(AnalysisCallback):
    features = ['word_count', 'star_rating', 'num_reviews']

    def __init__(self, feature: str, res: int = 5, **kwargs):
        super(PuzzlesResponse, self).__init__(**kwargs)
        assert feature in self.features, f"'{feature}' is not a valid feature"
        grid = np.meshgrid(np.linspace(0, 230, res), np.linspace(0, 5, res), np.linspace(0, 70, res))
        self.grid: pd.DataFrame = pd.DataFrame.from_dict({k: v.flatten() for k, v in zip(self.features, grid)})
        self.feature: str = feature

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration,
                        **additional_kwargs):
        self.grid[f'pred {iteration}'] = macs.predict(self.grid[self.features])

    def plot_function(self, iteration: Iteration) -> Optional[str]:
        fi, fj = [f for f in self.features if f != self.feature]
        li, ui = self.grid[fi].min(), self.grid[fi].max()
        lj, uj = self.grid[fj].min(), self.grid[fj].max()
        fader = ColorFader('black', 'magenta', 'cyan', 'yellow', bounds=[li, lj, ui, uj])
        for (i, j), group in self.grid.groupby([fi, fj]):
            label = f'{fi}: {i:.0f}, {fj}: {j:.0f}' if (i in [li, ui] and j in [lj, uj]) else None
            sns.lineplot(data=group, x=self.feature, y=f'pred {iteration}', color=fader(i, j), alpha=0.6, label=label)
        return f'{iteration}) {self.feature.replace("_", " ").upper()}'


# noinspection PyMissingOrEmptyDocstring
class RestaurantsAdjustment(AnalysisCallback):
    dollar_ratings = ['D', 'DD', 'DDD', 'DDDD']
    max_size = 40

    def __init__(self, rating: str, res: int = 100, data_points: bool = True, **kwargs):
        super(RestaurantsAdjustment, self).__init__(**kwargs)
        assert rating in self.dollar_ratings, f"'{rating}' is not a valid dollar rating"
        ar, nr = np.meshgrid(np.linspace(1, 5, res), np.linspace(0, 200, res))
        self.grid: pd.DataFrame = RestaurantsManager.process_data(pd.DataFrame.from_dict({
            'avg_rating': ar.flatten(),
            'num_reviews': nr.flatten(),
            'dollar_rating': [rating] * len(ar.flatten())
        }))[0]
        self.res: int = res
        self.rating: str = rating
        self.data_points: bool = data_points

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration,
                        **additional_kwargs):
        grid = self.grid[['avg_rating', 'num_reviews', 'D', 'DD', 'DDD', 'DDDD']]
        data = self.data[['avg_rating', 'num_reviews', 'D', 'DD', 'DDD', 'DDDD']]
        self.grid[f'pred {iteration}'] = macs.predict(grid)
        self.data[f'pred {iteration}'] = macs.predict(data)

    def on_adjustment_end(self, macs, x: Matrix, y: Vector, adjusted_y: Vector, val_data: Optional[Dataset],
                          iteration: Iteration, **additional_kwargs):
        self.data[f'adj {iteration}'] = adjusted_y
        self.data[f'sw {iteration}'] = additional_kwargs.get('sample_weight',
                                                             np.where(self.data['mask'] == 'label', 1, 0))

    def plot_function(self, iteration: Iteration) -> Optional[str]:
        # plot 3D response
        ctr = self.grid[f'pred {iteration}'].values.reshape(self.res, self.res)
        avg_ratings = self.grid['avg_rating'].values.reshape(self.res, self.res)
        num_reviews = self.grid['num_reviews'].values.reshape(self.res, self.res)
        plt.pcolor(avg_ratings, num_reviews, ctr, shading='auto', vmin=0, vmax=1)
        # plot data points
        if self.data_points:
            markers, sizes = AnalysisCallback.MARKERS, (0, RestaurantsAdjustment.max_size)
            sns.scatterplot(data=self.data, x='avg_rating', y='num_reviews', size=f'sw {iteration}', size_norm=(0, 1),
                            sizes=sizes, color='black', style='mask', markers=markers, legend=False)
        return f'{iteration}) {self.rating}'


# noinspection PyMissingOrEmptyDocstring
class SyntheticAdjustments2D(AnalysisCallback):
    max_size = 30
    alpha = 0.4

    def on_process_start(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], **additional_kwargs):
        super(SyntheticAdjustments2D, self).on_process_start(macs, x, y, val_data, **additional_kwargs)
        self.data['ground'] = SyntheticManager.function(self.data['a'], self.data['b'])

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration,
                        **additional_kwargs):
        self.data[f'pred {iteration}'] = macs.predict(x)
        self.data[f'pred err {iteration}'] = self.data[f'pred {iteration}'] - self.data['ground']

    def on_adjustment_end(self, macs, x: Matrix, y: Vector, adjusted_y: Vector, val_data: Optional[Dataset],
                          iteration: Iteration, **additional_kwargs):
        self.data[f'adj {iteration}'] = adjusted_y
        self.data[f'adj err {iteration}'] = self.data[f'adj {iteration}'] - self.data['ground']
        self.data[f'sw {iteration}'] = additional_kwargs.get('sample_weight',
                                                             np.where(self.data['mask'] == 'label', 1, 0))

    def plot_function(self, iteration: Iteration) -> Optional[str]:
        def synthetic_inverse(column):
            b = np.sin(np.pi * (self.data['b'] - 0.01)) ** 2 + 1
            return (self.data[column] - b) * b

        a, sw, pred = self.data['a'], self.data[f'sw {iteration}'].values, synthetic_inverse(f'pred {iteration}')
        s, m = self.data['mask'].values, AnalysisCallback.MARKERS
        ms, al = SyntheticAdjustments2D.max_size, SyntheticAdjustments2D.alpha
        sns.lineplot(x=self.data['a'], y=synthetic_inverse('ground'), color='green')
        sns.scatterplot(x=a, y=pred, color='red', alpha=al, s=ms)
        if iteration == AnalysisCallback.PRETRAINING:
            adj, color = synthetic_inverse('label'), 'black'
        else:
            adj, color = synthetic_inverse(f'adj {iteration}'), 'blue'
        sns.scatterplot(x=a, y=adj, style=s, markers=m, size=sw, size_norm=(0, 1), sizes=(0, ms), color=color, alpha=al)
        plt.legend(['ground', 'predictions', 'labels' if iteration == AnalysisCallback.PRETRAINING else 'adjusted'])
        return


# noinspection PyMissingOrEmptyDocstring
class SyntheticAdjustments3D(AnalysisCallback):
    max_size = 40

    def __init__(self, res: int = 100, data_points: bool = True, **kwargs):
        super(SyntheticAdjustments3D, self).__init__(**kwargs)
        assert self.sorting_attribute is None, 'sorting_attribute must be None'
        self.res: int = res
        self.data_points: bool = data_points
        self.val: Optional[pd.DataFrame] = None

    def on_process_start(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], **additional_kwargs):
        super(SyntheticAdjustments3D, self).on_process_start(macs, x, y, val_data, **additional_kwargs)
        # swap values and data in order to print the grid
        self.val = self.data.copy()
        a, b = np.meshgrid(np.linspace(-1, 1, self.res), np.linspace(-1, 1, self.res))
        self.data = pd.DataFrame.from_dict({'a': a.flatten(), 'b': b.flatten()})

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration,
                        **additional_kwargs):
        self.val[f'pred {iteration}'] = macs.predict(x)
        self.data[f'z {iteration}'] = macs.predict(self.data[['a', 'b']])

    def on_adjustment_end(self, macs, x: Matrix, y: Vector, adjusted_y: Vector, val_data: Optional[Dataset],
                          iteration: Iteration, **additional_kwargs):
        self.val[f'adj {iteration}'] = adjusted_y
        self.val[f'sw {iteration}'] = additional_kwargs.get('sample_weight',
                                                            np.where(self.val['mask'] == 'label', 1, 0))

    def plot_function(self, iteration: Iteration) -> Optional[str]:
        # plot 3D response
        ga = self.data['a'].values.reshape(self.res, self.res)
        gb = self.data['b'].values.reshape(self.res, self.res)
        gz = self.data[f'z {iteration}'].values.reshape(self.res, self.res)
        plt.pcolor(ga, gb, gz, shading='auto', cmap='viridis', vmin=gz.min(), vmax=gz.max())
        # plot data points
        if self.data_points:
            markers, sizes = AnalysisCallback.MARKERS, (0, SyntheticAdjustments3D.max_size)
            sns.scatterplot(data=self.val, x='a', y='b', size=f'sw {iteration}', size_norm=(0, 1), sizes=sizes,
                            color='black', style='mask', markers=markers, legend=False)
        plt.legend(['ground', 'label', 'adjusted'])
        return


# noinspection PyMissingOrEmptyDocstring
class SyntheticResponse(AnalysisCallback):
    def __init__(self, res: int = 10, **kwargs):
        super(SyntheticResponse, self).__init__(**kwargs)
        a, b = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
        self.grid = pd.DataFrame.from_dict({'a': a.flatten(), 'b': b.flatten()})
        self.fader = ColorFader('red', 'blue', bounds=[-1, 1])

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration,
                        **additional_kwargs):
        input_grid = self.grid[['a', 'b']]
        self.grid[f'pred {iteration}'] = macs.predict(input_grid)

    def plot_function(self, iteration: Iteration) -> Optional[str]:
        for idx, group in self.grid.groupby('b'):
            label = f'b = {idx:.0f}' if idx in [-1, 1] else None
            sns.lineplot(data=group, x='a', y=f'pred {iteration}', color=self.fader(idx), alpha=0.4, label=label)
        return
