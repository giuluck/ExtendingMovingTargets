import pandas as pd

from src.util import datasets
from src.restaurants import data

from src.moving_targets.learners import Learner, LinearRegression, LogisticRegression

if __name__ == '__main__':
    # train_data, val_data, test_data = datasets.load_data('res/restaurants.csv')
    # augmented_data = data.augment_data(train_data, n=5).drop(['index', 'monotonicity'], axis=1)
    # augmented_train_data = pd.concat((train_data, augmented_data)).reset_index(drop=True)
    # print(augmented_train_data)
    l = Learner()
    lrr = LinearRegression()
    lgr = LogisticRegression()
    print(l, lrr, lgr)
