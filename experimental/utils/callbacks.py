"""Moving Targets Callbacks."""

from typing import List, Dict, Union, Optional, Tuple

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
"""Data type for the output class or output info."""


class AnalysisCallback(Callback):
    """Template callback for results analysis and plotting."""

    PRETRAINING = 'PT'
    """Pretraining iteration name."""

    MARKERS = dict(aug='o', label='X')
    """Plot markers (a point for augmented data, a cross for original data)."""

    def __init__(self, num_columns: int = 5, sorting_attribute: object = None, file_signature: str = None,
                 do_plot: bool = True, **plt_kwargs):
        """
        :param num_columns:
            The number of columns of the subplot.

        :param sorting_attribute:
            Whether or not to sort attributes in a lexicographic order.

        :param file_signature:
            The filename and path where to store the output data, or None in case no output is required.

            The filename must have no extension, as two different files will be stored (a .csv and a .txt file).

        :param do_plot:
            Whether or not to show the final plot.

        :param plt_kwargs:
            Additional arguments for `plt()` function.
        """
        super(AnalysisCallback, self).__init__()

        self.num_columns: int = num_columns
        """The number of columns of the subplot."""

        self.sorting_attribute: object = sorting_attribute
        """Whether or not to sort attributes in a lexicographic order."""

        self.file_signature: str = file_signature
        """The filename and path where to store the output data, or None in case no output is required."""

        self.do_plot: bool = do_plot
        """Whether or not to show the final plot."""

        self.data: Optional[pd.DataFrame] = None
        """The dataframe of logged values which will be eventually plotted or stored in the output files."""

        self.iterations: List[Iteration] = []
        """A list of ordered iteration names."""

        self.plt_kwargs: Dict = {'figsize': (20, 10), 'tight_layout': True}
        """Default arguments for `plt()` function."""

        self.plt_kwargs.update(plt_kwargs)

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
                title = self._plot_function(it)
                ax.set(xlabel='', ylabel='')
                ax.set_title(f'{it})' if title is None else title)
            plt.show()

    def _plot_function(self, iteration: Iteration) -> Optional[str]:
        """Inner template method to build the plotting strategy.

        :param iteration:
            The iteration to plot.

        :return:
            An (optional) title of the subplot.
        """
        raise NotImplementedError("Please implement method '_plot_function'")


class DistanceAnalysis(AnalysisCallback):
    """Investigates the distance between ground truths, predictions, and the adjusted targets during iterations."""

    def __init__(self, ground_only: bool = True, num_columns=1, **plt_kwargs):
        """
        :param ground_only:
            Whether to consider ground data points only or augmented data points as well.

        :param num_columns:
            The number of columns of the subplot.

        :param plt_kwargs:
            Additional arguments for `plt()` function.
        """
        super(DistanceAnalysis, self).__init__(num_columns=num_columns, **plt_kwargs)

        self.ground_only: bool = ground_only
        """Whether to consider ground data points only or augmented data points as well."""

        self.y: Optional[str] = None
        """The name of the output feature."""

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

    def _plot_function(self, iteration: Iteration) -> Optional[str]:
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


class BoundsAnalysis(AnalysisCallback):
    """Investigates the lower and upper bounds for each data point during the master step."""

    def __init__(self, num_columns: int = 1, **plt_kwargs):
        """
        :param num_columns:
            The number of columns of the subplot.

        :param plt_kwargs:
            Additional arguments for `plt()` function.
        """
        super(BoundsAnalysis, self).__init__(num_columns=num_columns, **plt_kwargs)

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

    def _plot_function(self, iteration: Iteration) -> Optional[str]:
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


class CarsAdjustments(AnalysisCallback):
    """Investigates Moving Targets adjustments during iterations in cars dataset."""

    max_size = 50
    """Max size of markers."""

    alpha = 0.4
    """Alpha channel value."""

    def __init__(self, plot_kind: str = 'scatter', **plt_kwargs):
        """
        :param plot_kind:
            Either 'line' for a line output prediction, or 'scatter' for scatter output predictions.

        :param plt_kwargs:
            Additional arguments for `plt()` function.
        """
        super(CarsAdjustments, self).__init__(**plt_kwargs)
        assert plot_kind in ['line', 'scatter'], f"'{plot_kind}' is not a valid plot kind"

        self.plot_kind: str = plot_kind
        """Either 'line' for a line output prediction, or 'scatter' for scatter output predictions."""

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration,
                        **additional_kwargs):
        self.data[f'pred {iteration}'] = macs.predict(x)

    def on_adjustment_end(self, macs, x: Matrix, y: Vector, adjusted_y: Vector, val_data: Optional[Dataset],
                          iteration: Iteration, **additional_kwargs):
        self.data[f'adj {iteration}'] = adjusted_y
        self.data[f'sw {iteration}'] = additional_kwargs.get('sample_weight',
                                                             np.where(self.data['mask'] == 'label', 1, 0))

    def _plot_function(self, iteration: Iteration) -> Optional[str]:
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


class DefaultAdjustments(AnalysisCallback):
    """Investigates Moving Targets adjustments during iterations in default dataset."""

    max_size = 30
    """Max size of markers."""

    alpha = 0.8
    """Alpha channel value."""

    def __init__(self, **plt_kwargs):
        """
        :param plt_kwargs:
            Additional arguments for `plt()` function.
        """
        super(DefaultAdjustments, self).__init__(**plt_kwargs)
        married, payment = np.meshgrid([0, 1], np.arange(-2, 9))

        self.grid = pd.DataFrame.from_dict({'married': married.flatten(), 'payment': payment.flatten()})
        """The explicit grid used to plot the outputs."""

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration,
                        **additional_kwargs):
        self.data[f'pred {iteration}'] = macs.predict(x)
        self.grid[f'pred {iteration}'] = macs.predict(self.grid[['married', 'payment']])

    def on_adjustment_end(self, macs, x: Matrix, y: Vector, adjusted_y: Vector, val_data: Optional[Dataset],
                          iteration: Iteration, **additional_kwargs):
        self.data[f'adj {iteration}'] = adjusted_y
        self.data[f'sw {iteration}'] = additional_kwargs.get('sample_weight',
                                                             np.where(self.data['mask'] == 'label', 1, 0))

    def _plot_function(self, iteration: Iteration) -> Optional[str]:
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


class LawAdjustments(AnalysisCallback):
    """Investigates Moving Targets adjustments during iterations in law dataset."""

    max_size = 40
    """Max size of markers."""

    def __init__(self, res: int = 100, data_points: bool = True, **plt_kwargs):
        """
        :param res:
            The meshgrid resolution.

        :param data_points:
            Whether or not to plot data points over the mesh plot.

        :param plt_kwargs:
            Additional arguments for `plt()` function.
        """

        super(LawAdjustments, self).__init__(**plt_kwargs)
        lsat, ugpa = np.meshgrid(np.linspace(0, 50, res), np.linspace(0, 4, res))

        self.grid: pd.DataFrame = pd.DataFrame.from_dict({'lsat': lsat.flatten(), 'ugpa': ugpa.flatten()})
        """The explicit grid used to plot the outputs."""

        self.res: int = res
        """The meshgrid resolution."""

        self.data_points: bool = data_points
        """Whether or not to plot data points over the mesh plot."""

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

    def _plot_function(self, iteration: Iteration) -> Optional[str]:
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


class LawResponse(AnalysisCallback):
    """Investigates marginal feature responses during iterations in law dataset."""

    def __init__(self, feature: str, res: int = 100, **plt_kwargs):
        """
        :param feature:
            The feature for which to plot the response, either 'lsat' or 'ugpa'.

        :param res:
            The meshgrid resolution.

        :param plt_kwargs:
            Additional arguments for `plt()` function.
        """
        super(LawResponse, self).__init__(**plt_kwargs)
        assert feature in ['lsat', 'ugpa'], f"'{feature}' is not a valid feature"
        lsat, ugpa = np.meshgrid(np.linspace(0, 50, res), np.linspace(0, 4, res))

        self.grid: pd.DataFrame = pd.DataFrame.from_dict({'lsat': lsat.flatten(), 'ugpa': ugpa.flatten()})
        """The explicit grid used to plot the outputs."""

        self.fader: ColorFader = ColorFader('red', 'blue', bounds=[0, 4] if feature == 'lsat' else [0, 50])
        """The `ColorFader` object to show the responses."""

        self.features: Tuple[str, str] = ('lsat', 'ugpa') if feature == 'lsat' else ('ugpa', 'lsat')
        """A tuple containing the two monotonic features, where the first one is the investigated feature while the
        second one is used for grouping."""

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration,
                        **additional_kwargs):
        input_grid = self.grid[['lsat', 'ugpa']]
        self.grid[f'pred {iteration}'] = macs.predict(input_grid)

    def _plot_function(self, iteration: Iteration) -> Optional[str]:
        feat, group_feat = self.features
        for group_val, group in self.grid.groupby([group_feat]):
            sns.lineplot(data=group, x=feat, y=f'pred {iteration}', color=self.fader(group_val), alpha=0.6)
        return f'{iteration}) {feat.upper()}'


class PuzzlesResponse(AnalysisCallback):
    """Investigates marginal feature responses during iterations in puzzles dataset."""

    features = ['word_count', 'star_rating', 'num_reviews']
    """Puzzles dataset monotonic features."""

    def __init__(self, feature: str, res: int = 5, **plt_kwargs):
        """
        :param feature:
            The feature for which to plot the response, either 'word_count', 'star_rating', or 'num_reviews'.

        :param res:
            The meshgrid resolution.

        :param plt_kwargs:
            Additional arguments for `plt()` function.
        """
        super(PuzzlesResponse, self).__init__(**plt_kwargs)
        assert feature in self.features, f"'{feature}' is not a valid feature"
        grid = np.meshgrid(np.linspace(0, 230, res), np.linspace(0, 5, res), np.linspace(0, 70, res))

        self.grid: pd.DataFrame = pd.DataFrame.from_dict({k: v.flatten() for k, v in zip(self.features, grid)})
        """The explicit grid used to plot the outputs."""

        self.feature: str = feature
        """The feature for which to plot the response, either 'word_count', 'star_rating', or 'num_reviews'."""

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration,
                        **additional_kwargs):
        self.grid[f'pred {iteration}'] = macs.predict(self.grid[self.features])

    def _plot_function(self, iteration: Iteration) -> Optional[str]:
        fi, fj = [f for f in self.features if f != self.feature]
        li, ui = self.grid[fi].min(), self.grid[fi].max()
        lj, uj = self.grid[fj].min(), self.grid[fj].max()
        fader = ColorFader('black', 'magenta', 'cyan', 'yellow', bounds=[li, lj, ui, uj])
        for (i, j), group in self.grid.groupby([fi, fj]):
            label = f'{fi}: {i:.0f}, {fj}: {j:.0f}' if (i in [li, ui] and j in [lj, uj]) else None
            sns.lineplot(data=group, x=self.feature, y=f'pred {iteration}', color=fader(i, j), alpha=0.6, label=label)
        return f'{iteration}) {self.feature.replace("_", " ").upper()}'


class RestaurantsAdjustment(AnalysisCallback):
    """Investigates Moving Targets adjustments during iterations in restaurants dataset."""

    dollar_ratings = ['D', 'DD', 'DDD', 'DDDD']
    """The four possible categories of dollar ratings."""

    max_size = 40
    """Max size of markers."""

    def __init__(self, rating: str, res: int = 100, data_points: bool = True, **plt_kwargs):
        """
        :param rating:
            The dollar rating to investigate.

        :param res:
            The meshgrid resolution.

        :param data_points:
            Whether or not to plot data points over the mesh plot.

        :param plt_kwargs:
            Additional arguments for `plt()` function.
        """
        super(RestaurantsAdjustment, self).__init__(**plt_kwargs)
        assert rating in self.dollar_ratings, f"'{rating}' is not a valid dollar rating"
        ar, nr = np.meshgrid(np.linspace(1, 5, res), np.linspace(0, 200, res))

        self.grid: pd.DataFrame = RestaurantsManager.process_data(pd.DataFrame.from_dict({
            'avg_rating': ar.flatten(),
            'num_reviews': nr.flatten(),
            'dollar_rating': [rating] * len(ar.flatten())
        }))[0]
        """The explicit grid used to plot the outputs."""

        self.res: int = res
        """The meshgrid resolution."""

        self.rating: str = rating
        """The dollar rating to investigate."""

        self.data_points: bool = data_points
        """Whether or not to plot data points over the mesh plot."""

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

    def _plot_function(self, iteration: Iteration) -> Optional[str]:
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


class SyntheticAdjustments2D(AnalysisCallback):
    """Investigates Moving Targets adjustments (on the monotonic feature) during iterations in synthetic dataset."""

    max_size = 30
    """Max size of markers."""

    alpha = 0.4
    """Alpha channel value."""

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

    def _plot_function(self, iteration: Iteration) -> Optional[str]:
        def synthetic_inverse(column):
            """Computes the value of the expected value of the 'a' feature given the output label.

            :param column:
                The output column (it can be the ground truth, a prediction, ...).

            :return:
                The expected value of the 'a' feature given the output label.
            """
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


class SyntheticAdjustments3D(AnalysisCallback):
    """Investigates Moving Targets adjustments (on both the features) during iterations in synthetic dataset."""

    max_size = 40
    """Max size of markers."""

    def __init__(self, res: int = 100, data_points: bool = True, **plt_kwargs):
        """
        :param res:
            The meshgrid resolution.

        :param data_points:
            Whether or not to plot data points over the mesh plot.

        :param plt_kwargs:
            Additional arguments for `plt()` function.
        """
        super(SyntheticAdjustments3D, self).__init__(**plt_kwargs)
        assert self.sorting_attribute is None, 'sorting_attribute must be None'

        self.res: int = res
        """The meshgrid resolution."""

        self.data_points: bool = data_points
        """Whether or not to plot data points over the mesh plot."""

        self.val: Optional[pd.DataFrame] = None
        """The data values to be plotted."""

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

    def _plot_function(self, iteration: Iteration) -> Optional[str]:
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


class SyntheticResponse(AnalysisCallback):
    """Investigates marginal feature responses during iterations in synthetic dataset."""

    def __init__(self, res: int = 10, **plt_kwargs):
        """
        :param res:
            The meshgrid resolution.

        :param plt_kwargs:
            Additional arguments for `plt()` function.
        """
        super(SyntheticResponse, self).__init__(**plt_kwargs)
        a, b = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))

        self.grid = pd.DataFrame.from_dict({'a': a.flatten(), 'b': b.flatten()})
        """The explicit grid used to plot the outputs."""

        self.fader = ColorFader('red', 'blue', bounds=[-1, 1])
        """The `ColorFader` object to show the responses."""

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration,
                        **additional_kwargs):
        input_grid = self.grid[['a', 'b']]
        self.grid[f'pred {iteration}'] = macs.predict(input_grid)

    def _plot_function(self, iteration: Iteration) -> Optional[str]:
        for idx, group in self.grid.groupby('b'):
            label = f'b = {idx:.0f}' if idx in [-1, 1] else None
            sns.lineplot(data=group, x='a', y=f'pred {iteration}', color=self.fader(idx), alpha=0.4, label=label)
        return
