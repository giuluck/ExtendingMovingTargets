import os
os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

from src import restaurants
from src.moving_targets.callbacks import WandBLogger
from src.util.comb import cartesian_product
from src.util.preprocessing import Scaler
from src.moving_targets.metrics import AUC
from src.restaurants.models import MLP, MTLearner, MTMaster, MT
from src.restaurants import compute_monotonicities

def get_monotonicities_list(data, kind):
    higher_indices, lower_indices = [], []
    if kind == 'ground':
        for idx, rec in data.iterrows():
            monotonicity, ground_index = rec['monotonicity'], int(rec['ground_index'])
            if monotonicity != 0:
                higher_indices.append(idx if monotonicity > 0 else ground_index)
                lower_indices.append(ground_index if monotonicity > 0 else idx)
    elif kind == 'group':
        for index, group in data.groupby('ground_index'):
            values = group.drop(['clicked', 'ground_index', 'monotonicity'], axis=1).values
            his, lis = np.where(compute_monotonicities(values, values) == 1)
            higher_indices.append(group.index.values[his])
            lower_indices.append(group.index.values[lis])
        higher_indices = np.concatenate(higher_indices)
        lower_indices = np.concatenate(lower_indices)
    elif kind == 'all':
        values = data.drop(['clicked', 'ground_index', 'monotonicity'], axis=1).values
        higher_indices, lower_indices = np.where(compute_monotonicities(values, values) == 1)
    return [(hi, li) for hi, li in zip(higher_indices, lower_indices)]


def build_model(h_units, scaler):
    model = MLP(output_act='sigmoid', h_units=h_units, scaler=scaler)
    model.compile(optimizer='adam', loss='mse')
    return model


if __name__ == '__main__':
    # load and prepare data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = restaurants.load_data()
    aug_data, aug_info = restaurants.augment_data(x_train, n=5)
    x_aug = pd.concat((x_train, aug_data)).reset_index(drop=True)
    y_aug = pd.concat((y_train, aug_info)).rename({0: 'clicked'}, axis=1).reset_index(drop=True)
    y_aug = y_aug.fillna({'ground_index': pd.Series(y_aug.index), 'clicked': -1, 'monotonicity': 0}).astype('int')
    aug_scaler = Scaler(x_aug, methods=dict(avg_rating='std', num_reviews='std'))
    x, y = x_aug.values, y_aug['clicked'].values

    # compute monotonicity list for each of the three possibilities
    full_data = pd.concat((x_aug, y_aug), axis=1)
    monotonicities = {k: get_monotonicities_list(full_data, k) for k in ['ground', 'group', 'all']}

    # create study list
    study = cartesian_product(
        h_units=[[], [16, 8, 8]],
        restart_fit=[True, False],
        alpha=list(np.logspace(-2, 2, 13)),
        beta=[1.],
        monotonicities=['ground', 'group', 'all']
    )

    # begin study
    for i, params in enumerate(study):
        start_time = time.time()
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end='')
        mt = MT(
            learner=MTLearner(
                build_model=lambda: build_model(params['h_units'], aug_scaler),
                restart_fit=params['restart_fit'],
                validation_data=(x_val, y_val),
                callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                epochs=200,
                verbose=0
            ),
            master=MTMaster(monotonicities[params['monotonicities']], alpha=params['alpha'], beta=params['beta']),
            evaluation_data=dict(train=(x_train, y_train), val=(x_val, y_val), test=(x_test, y_test)),
            metrics=[AUC(name='auc')]
        )
        mt.fit(x, y, iterations=10, callbacks=[WandBLogger('shape_constraints', 'giuluck', 'restaurants', **params)])
        print(f' -- elapsed time: {time.time() - start_time}')
