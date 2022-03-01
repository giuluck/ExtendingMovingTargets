from typing import Dict, Callable, Any, Optional, Union, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from moving_targets.util.scalers import Scaler

from src.datasets.manager import Manager, AnalysisCallback


class Restaurants(Manager):
    """Data Manager for the Restaurants Dataset."""

    __name__: str = 'restaurants'

    callbacks: Dict[str, Callable] = {
        **Manager.callbacks,
        'response_D': lambda fs: RestaurantsResponse(rating='D', file_signature=fs),
        'response_DD': lambda fs: RestaurantsResponse(rating='DD', file_signature=fs),
        'response_DDD': lambda fs: RestaurantsResponse(rating='DDD', file_signature=fs),
        'response_DDDD': lambda fs: RestaurantsResponse(rating='DDDD', file_signature=fs)
    }

    @staticmethod
    def ctr(avg_ratings, num_reviews, dollar_ratings) -> Any:
        """Computes the click-trough rate estimate."""
        dollar_rating_baseline = {'D': 3, 'DD': 2, 'DDD': 4, 'DDDD': 4.5}
        dollar_ratings = np.reshape([dollar_rating_baseline[d] for d in dollar_ratings.flatten()], dollar_ratings.shape)
        return 1 / (1 + np.exp(dollar_ratings - avg_ratings * np.log1p(num_reviews) / 4))

    @staticmethod
    def onehot(df: pd.DataFrame) -> pd.DataFrame:
        """Onehot encodes the categorical feature."""
        df = df.copy()
        for rating in ['DDDD', 'DDD', 'DD', 'D']:
            df.insert(2, rating, (df['dollar_rating'] == rating).astype('float'))
        return df.drop('dollar_rating', axis=1)

    @staticmethod
    def sample(n: int, testing_set: bool = True) -> pd.DataFrame:
        """Sample data points from the restaurants. Depending on the value of 'testing_set', samples either from the
        train or test distribution."""
        rng = np.random.default_rng(seed=0)
        # sample restaurants and compute ctr estimates
        avg_ratings = rng.uniform(1.0, 5.0, n)
        num_reviews = np.round(np.exp(rng.uniform(0.0, np.log(200), n)))
        dollar_ratings = rng.choice(['D', 'DD', 'DDD', 'DDDD'], n)
        ctr_labels = Restaurants.ctr(avg_ratings, num_reviews, dollar_ratings)
        # then sample multiple records for each restaurant with target clicked/not clicked
        if testing_set:
            # the test set has a more uniform distribution over all restaurants
            num_views = rng.poisson(lam=3, size=n)
        else:
            # while the training set has more views on popular restaurants
            num_views = rng.poisson(lam=ctr_labels * num_reviews / 50.0, size=n)
        # build dataframe and onehot encode dollar rating
        df = pd.DataFrame({
            'avg_rating': np.repeat(avg_ratings, num_views),
            'num_reviews': np.repeat(num_reviews, num_views),
            'dollar_rating': np.repeat(dollar_ratings, num_views),
            'clicked': rng.binomial(n=1, p=np.repeat(ctr_labels, num_views)).astype('float')
        })
        return Restaurants.onehot(df)

    @classmethod
    def load(cls) -> Dict[str, pd.DataFrame]:
        return {'train': Restaurants.sample(1000, testing_set=False), 'test': Restaurants.sample(600, testing_set=True)}

    @classmethod
    def grid(cls, plot: bool = True) -> pd.DataFrame:
        if plot:
            ar, nr = np.meshgrid(np.linspace(1, 5, 100), np.linspace(0, 200, 100))
            grid = pd.DataFrame.from_dict({'avg_rating': ar.flatten(), 'num_reviews': nr.flatten()})
            for rating in ['D', 'DD', 'DDD', 'DDDD']:
                grid[rating] = 0
            return grid
        else:
            ar, nr, dr = np.meshgrid(np.linspace(1, 5, 30), np.linspace(0, 200, 30), ['D', 'DD', 'DDD', 'DDDD'])
            grid = pd.DataFrame.from_dict({
                'avg_rating': ar.flatten(),
                'num_reviews': nr.flatten(),
                'dollar_rating': dr.flatten()
            })
            return Restaurants.onehot(grid)

    def __init__(self):
        super(Restaurants, self).__init__(label='clicked',
                                          directions={'avg_rating': 1, 'num_reviews': 1},
                                          classification=True)

    def scalers(self) -> Tuple[Optional[Scaler], Optional[Scaler]]:
        return Scaler(default_method='none', avg_rating='std', num_reviews='std'), None

    def _plot(self, model):
        grid = self.grid(plot=True)
        res = np.sqrt(len(grid)).astype(int)
        ar, nr = grid['avg_rating'].values.reshape((res, res)), grid['num_reviews'].values.reshape((res, res))
        _, axes = plt.subplots(2, 4, sharex='all', sharey='all', figsize=(16, 9), tight_layout=True)
        axes = axes.reshape((2, 4))
        for i, rating in enumerate(['D', 'DD', 'DDD', 'DDDD']):
            # compute true and predicted response
            df = grid.copy()
            df[rating] = 1
            pred = model.predict(df).reshape((res, res))
            ctr = Restaurants.ctr(avg_ratings=ar, num_reviews=nr, dollar_ratings=np.array([[rating] * res] * res))
            # plot responses
            axes[0, i].set(title=rating, xlabel=None, ylabel='Real CTR' if i == 0 else None)
            axes[0, i].pcolor(ar, nr, ctr, shading='auto', cmap='viridis', vmin=0, vmax=1)
            axes[1, i].set(xlabel=None, ylabel='Estimated CTR' if i == 0 else None)
            axes[1, i].pcolor(ar, nr, pred, shading='auto', cmap='viridis', vmin=0, vmax=1)


class RestaurantsResponse(AnalysisCallback):
    """Investigates features response during iterations in restaurants dataset."""

    ratings = ['D', 'DD', 'DDD', 'DDDD']

    max_size = 40
    """Max size of markers."""

    def __init__(self,
                 rating: str,
                 sorting_attribute: Optional[str] = None,
                 file_signature: Optional[str] = None,
                 num_columns: Union[int, str] = 'auto'):
        super(RestaurantsResponse, self).__init__(sorting_attribute=sorting_attribute,
                                                  file_signature=file_signature,
                                                  num_columns=num_columns)
        assert rating in self.ratings, f"'{rating}' is not a dollar rating"
        self.rating: str = rating
        self.data: pd.DataFrame = Restaurants.grid(plot=True)
        self.data[rating] = 1

    def on_training_end(self, macs, x, y, p, val_data):
        data = self.data[['avg_rating', 'num_reviews', 'D', 'DD', 'DDD', 'DDDD']]
        self.data[f'pred {macs.iteration}'] = macs.predict(data)

    def _plot_function(self, iteration: Any) -> Optional[str]:
        # plot 3D response
        res = np.sqrt(len(self.data)).astype(int)
        ctr = self.data[f'pred {iteration}'].values.reshape(res, res)
        avg_ratings = self.data['avg_rating'].values.reshape(res, res)
        num_reviews = self.data['num_reviews'].values.reshape(res, res)
        plt.pcolor(avg_ratings, num_reviews, ctr, shading='auto', vmin=0, vmax=1)
        return f'{iteration}) {self.rating}'
