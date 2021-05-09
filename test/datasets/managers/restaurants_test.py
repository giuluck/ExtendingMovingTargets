import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from moving_targets.metrics import CrossEntropy, AUC
from src.datasets import RestaurantsManager
from src.models import MTClassificationMaster
from test.datasets.managers.test_manager import TestManager, AnalysisCallback


class RestaurantsTest(TestManager):
    def __init__(self, warm_start=False, **kwargs):
        super(RestaurantsTest, self).__init__(
            dataset=RestaurantsManager(),
            master_type=MTClassificationMaster,
            metrics=[CrossEntropy(), AUC()],
            data_args=dict(),
            augmented_args=dict(num_augmented=5),
            monotonicities_args=dict(kind='group'),
            learner_args=dict(output_act='sigmoid', h_units=[16, 8, 8], optimizer='adam', loss='binary_crossentropy',
                              warm_start=warm_start),
            **kwargs
        )


class RestaurantsResponse(AnalysisCallback):
    dollar_ratings = ['D', 'DD', 'DDD', 'DDDD']

    def __init__(self, rating, res=100, **kwargs):
        super(RestaurantsResponse, self).__init__(**kwargs)
        assert rating in self.dollar_ratings, f"rating should be in {self.dollar_ratings}"
        assert self.sorting_attribute is None, "'sorting_attribute' field must be None"
        ar, nr = np.meshgrid(np.linspace(1, 5, res), np.linspace(0, 200, res))
        self.grid, _ = RestaurantsManager.process_data(pd.DataFrame.from_dict({
            'avg_rating': ar.flatten(),
            'num_reviews': nr.flatten(),
            'dollar_rating': [rating] * len(ar.flatten())
        }))
        self.res = res
        self.rating = rating

    def on_training_end(self, macs, x, y, val_data, iteration, **kwargs):
        x = self.grid[['avg_rating', 'num_reviews', 'D', 'DD', 'DDD', 'DDDD']]
        self.grid[f'pred {iteration}'] = macs.predict(x)

    def plot_function(self, iteration):
        ctr = self.grid[f'pred {iteration}'].values.reshape(self.res, self.res)
        avg_ratings = self.grid['avg_rating'].values.reshape(self.res, self.res)
        num_reviews = self.grid['num_reviews'].values.reshape(self.res, self.res)
        plt.pcolor(avg_ratings, num_reviews, ctr, shading='auto', vmin=0, vmax=1)
        return f'{iteration}) {self.rating}'
