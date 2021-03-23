import pandas as pd
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from sklearn.metrics import roc_auc_score

from src.restaurants import data, plot


class MLPBatchGenerator(Sequence):
    def __init__(self, df, batch_size):
        super(MLPBatchGenerator).__init__()
        df = df.reset_index(drop=True)
        df['avg_rating'] = (df['avg_rating'] - df['avg_rating'].mean()) / df['avg_rating'].std()
        df['num_reviews'] = (df['num_reviews'] - df['num_reviews'].mean()) / df['num_reviews'].std()
        df['D'] = (df['dollar_rating'] == 'D').astype('float')
        df['DD'] = (df['dollar_rating'] == 'DD').astype('float')
        df['DDD'] = (df['dollar_rating'] == 'DDD').astype('float')
        df['DDDD'] = (df['dollar_rating'] == 'DDDD').astype('float')
        df['btc'] = df.index // batch_size
        self.batches = [b.drop(['btc', 'dollar_rating'], axis=1).reset_index(drop=True) for _, b in df.groupby('btc')]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        batch: pd.DataFrame = self.batches[index]
        x = batch[['avg_rating', 'num_reviews', 'D', 'DD', 'DDD', 'DDDD']]
        y = batch['clicked']
        return x.values, y.values


class MLPClassifier(Model):
    def __init__(self, hidden=None):
        super(MLPClassifier, self).__init__()
        self.lrs = [] if hidden is None else [Dense(h, activation='relu') for h in hidden]
        self.lrs = self.lrs + [Dense(1, activation='sigmoid')]

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.lrs:
            x = layer(x)
        return x

    def ctr_estimate(self, avg_rating, num_reviews, dr):
        df = pd.DataFrame({
            'avg_rating': avg_rating,
            'num_reviews': num_reviews,
            'dollar_rating': dr,
            'clicked': [-1] * len(avg_rating)
        })
        return self.predict(MLPBatchGenerator(df, 32))

    def evaluation_summary(self, train_data, val_data, test_data, figsize=(14, 3)):
        train_predictions = self.predict(MLPBatchGenerator(train_data, 32))
        val_predictions = self.predict(MLPBatchGenerator(val_data, 32))
        test_predictions = self.predict(MLPBatchGenerator(test_data, 32))
        print(roc_auc_score(train_data['clicked'], train_predictions), '(train auc)', end=', ')
        print(roc_auc_score(val_data['clicked'], val_predictions), '(val auc)', end=', ')
        print(roc_auc_score(test_data['clicked'], test_predictions), '(test auc)')
        plot.click_through_rate(self.ctr_estimate, title='Estimated CTR', figsize=figsize)
        plot.click_through_rate(data.ctr_estimate, title='Real CTR', figsize=figsize)
