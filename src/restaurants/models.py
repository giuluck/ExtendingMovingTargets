import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score
from tensorflow.keras.utils import to_categorical

from src import restaurants
from src.models import MLP, SBR, MT


class RestaurantsModel:
    def __init__(self):
        self.avg_ratings = self.num_reviews = self.dollar_ratings = self.ground_truths = None

    def initialize(self):
        ar, nr, dr = np.meshgrid(np.linspace(1, 5, num=100), np.linspace(0, 200, num=100), ['D', 'DD', 'DDD', 'DDDD'])
        self.avg_ratings = ar.reshape(-1, )
        self.num_reviews = nr.reshape(-1, )
        self.dollar_ratings = dr.reshape(-1, )
        self.ground_truths = restaurants.ctr_estimate(self.avg_ratings, self.num_reviews, self.dollar_ratings)

    def predict(self, x):
        raise NotImplementedError("Please implement method 'predict'")

    def ctr_estimate(self, avg_rating, num_reviews, dollar_rating):
        df = pd.DataFrame({'avg_rating': avg_rating, 'num_reviews': num_reviews})
        df[['D', 'DD', 'DDD', 'DDDD']] = to_categorical([len(dr) - 1 for dr in dollar_rating], num_classes=4)
        return self.predict(df)

    def compute_ground_r2(self):
        pred = self.ctr_estimate(self.avg_ratings, self.num_reviews, self.dollar_ratings)
        return r2_score(self.ground_truths, pred)

    def evaluation_summary(self, figsize=(14, 3), **kwargs):
        for title, (x, y) in kwargs.items():
            print(f'{roc_auc_score(y, self.predict(x)):.4} ({title} auc)', end=', ')
        print(f'{self.compute_ground_r2():.4} (ground r2)')
        restaurants.plot_ctr(self.ctr_estimate, title='Estimated CTR', figsize=figsize)
        restaurants.plot_ctr(restaurants.ctr_estimate, title='Real CTR', figsize=figsize)


class RestaurantsMLP(MLP, RestaurantsModel):
    def __init__(self, output_act, h_units=None, scaler=None):
        super(RestaurantsMLP, self).__init__(output_act=output_act, h_units=h_units, scaler=scaler)
        super(RestaurantsMLP, self).initialize()


class RestaurantsSBR(SBR, RestaurantsModel):
    def __init__(self, output_act, h_units=None, scaler=None, alpha=None, regularizer_act=None):
        super(RestaurantsSBR, self).__init__(output_act=output_act,
                                             h_units=h_units,
                                             scaler=scaler,
                                             alpha=alpha,
                                             regularizer_act=regularizer_act)
        super(RestaurantsSBR, self).initialize()


class RestaurantsMT(MT, RestaurantsModel):
    def __init__(self, learner, master, init_step='pretraining', metrics=None):
        super(RestaurantsMT, self).__init__(learner=learner, master=master, init_step=init_step, metrics=metrics)
        super(RestaurantsMT, self).initialize()

    def on_iteration_end(self, macs, x, y, val_data, iteration):
        super().on_iteration_end(macs, x, y, val_data, iteration)
        self.log(**{'learner/ground_r2': self.compute_ground_r2()})
