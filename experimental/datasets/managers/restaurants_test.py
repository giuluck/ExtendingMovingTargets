"""Restaurants Test Manager & Callbacks."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional as Opt

from moving_targets.metrics import AUC
from moving_targets.util.typing import Iteration, Vector, Matrix, Dataset
from src.datasets import RestaurantsManager
from experimental.datasets.managers.test_manager import ClassificationTest, AnalysisCallback


# noinspection PyMissingOrEmptyDocstring
class RestaurantsTest(ClassificationTest):
    def __init__(self, lrn_h_units: tuple = (16, 8, 8), **kwargs):
        super(RestaurantsTest, self).__init__(
            dataset=RestaurantsManager(),
            lrn_h_units=lrn_h_units,
            mst_evaluation_metric=AUC(name='metric'),
            **kwargs
        )


# noinspection PyMissingOrEmptyDocstring
class RestaurantsAdjustment(AnalysisCallback):
    dollar_ratings = ['D', 'DD', 'DDD', 'DDDD']
    max_size = 40

    def __init__(self, rating: str, res: int = 100, data_points: bool = True, **kwargs):
        super(RestaurantsAdjustment, self).__init__(**kwargs)
        assert rating in self.dollar_ratings, f"rating should be in {self.dollar_ratings}"
        ar, nr = np.meshgrid(np.linspace(1, 5, res), np.linspace(0, 200, res))
        self.grid: pd.DataFrame = RestaurantsManager.process_data(pd.DataFrame.from_dict({
            'avg_rating': ar.flatten(),
            'num_reviews': nr.flatten(),
            'dollar_rating': [rating] * len(ar.flatten())
        }))[0]
        self.res: int = res
        self.rating: str = rating
        self.data_points: bool = data_points

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        grid = self.grid[['avg_rating', 'num_reviews', 'D', 'DD', 'DDD', 'DDDD']]
        data = self.data[['avg_rating', 'num_reviews', 'D', 'DD', 'DDD', 'DDDD']]
        self.grid[f'pred {iteration}'] = macs.predict(grid)
        self.data[f'pred {iteration}'] = macs.predict(data)

    def on_adjustment_end(self, macs, x: Matrix, y: Vector, adjusted_y: Vector, val_data: Opt[Dataset],
                          iteration: Iteration, **kwargs):
        self.data[f'adj {iteration}'] = adjusted_y
        self.data[f'sw {iteration}'] = kwargs.get('sample_weight', np.where(self.data['mask'] == 'label', 1, 0))

    def plot_function(self, iteration: Iteration) -> Opt[str]:
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
