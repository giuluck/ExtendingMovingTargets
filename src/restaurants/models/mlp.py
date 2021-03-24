import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_auc_score

from src import restaurants


class MLPClassifier(Model):
    def __init__(self, hidden=None, scaler=None):
        super(MLPClassifier, self).__init__()
        self.scaler = scaler
        self.lrs = [] if hidden is None else [Dense(h, activation='relu') for h in hidden]
        self.lrs = self.lrs + [Dense(1, activation='sigmoid')]

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        x = inputs if self.scaler is None else self.scaler.transform(inputs)
        for layer in self.lrs:
            x = layer(x)
        return x

    def ctr_estimate(self, avg_rating, num_reviews, dollar_rating):
        df = pd.DataFrame({'avg_rating': avg_rating, 'num_reviews': num_reviews})
        df[['D', 'DD', 'DDD', 'DDDD']] = to_categorical([len(dr) - 1 for dr in dollar_rating], num_classes=4)
        return self.predict(df)

    def evaluation_summary(self, figsize=(14, 3), **kwargs):
        summary = ', '.join([f'{roc_auc_score(y, self.predict(x))} ({title} auc)' for title, (x, y) in kwargs.items()])
        print(summary)
        restaurants.plot_ctr(self.ctr_estimate, title='Estimated CTR', figsize=figsize)
        restaurants.plot_ctr(restaurants.ctr_estimate, title='Real CTR', figsize=figsize)
