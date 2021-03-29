import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

from src import restaurants
from src.util.preprocessing import Scaler
from src.restaurants import compute_monotonicities, models


def get_monotonicities_list(data):
    higher_indices, lower_indices = [], []
    for index, group in data.groupby('ground_index'):
        values = group.drop('ground_index', axis=1).values
        his, lis = np.where(compute_monotonicities(values, values) == 1)
        higher_indices.append(group.index.values[his])
        lower_indices.append(group.index.values[lis])
    higher_indices = np.concatenate(higher_indices)
    lower_indices = np.concatenate(lower_indices)
    return [(hi, li) for hi, li in zip(higher_indices, lower_indices)]


if __name__ == '__main__':
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = restaurants.load_data()
    n = len(x_train)

    aug_data, aug_info = restaurants.augment_data(x_train.iloc[:n], n=5)
    x_aug = pd.concat((x_train.iloc[:n], aug_data)).reset_index(drop=True)
    y_aug = pd.concat((y_train.iloc[:n], aug_info)).rename({0: 'clicked'}, axis=1).reset_index(drop=True)
    y_aug = y_aug.fillna({'ground_index': pd.Series(y_aug.index), 'clicked': -1, 'monotonicity': 0}).astype('int')
    aug_scaler = Scaler(x_aug, methods=dict(avg_rating='std', num_reviews='std'))

    # pd.set_option('display.max_rows', 100)
    # pd.set_option('display.max_columns', 100)
    # pd.set_option('display.width', 1000)
    # data = pd.DataFrame(xag.values[:, :2], columns=['avg_rating', 'num_reviews'])
    # data['dollar_rating'] = ['D' * (dr + 1) for dr in xag.values[:, 2:6].argmax(axis=1)]
    # data['clicked'] = yag.values[:, 0]

    mono = get_monotonicities_list(pd.concat((x_aug, y_aug['ground_index']), axis=1))
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    fit_args = dict(epochs=200, validation_data=(x_val, y_val), callbacks=callbacks, verbose=1)
    mt = models.MT(monotonicities=mono, h_units=[16, 8, 8], scaler=aug_scaler, alpha=1.0, restart_fit=True, **fit_args)
    mt.fit(x_aug.values, y_aug['clicked'].values, iterations=5)

    print('TRAIN ->', mt.evaluate(x_train, y_train))
    print('VAL ->', mt.evaluate(x_val, y_val))
    print('TEST ->', mt.evaluate(x_test, y_test))
    print()

    mt.learner.model.evaluation_summary(train=(x_train, y_train), val=(x_val, y_val), test=(x_test, y_test))
    plt.show()
