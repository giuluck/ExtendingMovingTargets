import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_auc_score, r2_score

from src import restaurants

class Model:
    def __init__(self):
        super(Model, self).__init__()
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
