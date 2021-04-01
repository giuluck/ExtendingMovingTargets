import os
os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import shutil
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from src import restaurants
from moving_targets.callbacks import WandBLogger
from moving_targets.metrics import AUC
from src.models import MTLearner, MTMaster, MT
from src.restaurants import compute_monotonicities
from src.restaurants.models import RestaurantsMLP
from src.util.augmentation import augment_data
from src.util.combinatorial import cartesian_product
from src.util.preprocessing import Scaler

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


def get_model(h_units, scaler):
    model = RestaurantsMLP(output_act='sigmoid', h_units=h_units, scaler=scaler)
    model.compile(optimizer='adam', loss='mse')
    return model


def get_augmented_data(xtr, ytr, ns=None, n=5):
    if ns is not None:
        xtr = xtr.iloc[:ns]
        ytr = ytr.iloc[:ns]
    agd, agi = augment_data(xtr, n=n, compute_monotonicities=compute_monotonicities, sampling_functions={
        'avg_rating': lambda s: np.random.uniform(1.0, 5.0, size=s),
        'num_reviews': lambda s: np.round(np.exp(np.random.uniform(0.0, np.log(200), size=s))),
        ('D', 'DD', 'DDD', 'DDDD'): lambda s: to_categorical(np.random.randint(4, size=s), num_classes=4)
    })
    xag = pd.concat((xtr, agd)).reset_index(drop=True)
    yag = pd.concat((ytr, agi)).rename({0: 'clicked'}, axis=1).reset_index(drop=True)
    yag = yag.fillna({'ground_index': pd.Series(yag.index), 'monotonicity': 0})
    scl = Scaler(xag, methods=dict(avg_rating='std', num_reviews='std'))
    fag = pd.concat((xag, yag), axis=1)
    return xag, yag, fag, scl


if __name__ == '__main__':
    # set random seeds
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    # load and prepare data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = restaurants.load_data()
    x_aug, y_aug, full_aug, aug_scaler = get_augmented_data(x_train, y_train)
    monotonicities = {k: get_monotonicities_list(full_aug, k) for k in ['ground', 'group', 'all']}

    # create study list
    study = cartesian_product(
        # init_step=['pretraining', 'projection'],
        h_units=[[16, 8, 8]],
        restart_fit=[True],
        alpha=[0.1],
        beta=[0.1],
        monotonicities=['group']
    )

    # begin study
    for i, params in enumerate(study):
        start_time = time.time()
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end='')
        mt = MT(
            learner=MTLearner(
                build_model=lambda: get_model(params['h_units'], aug_scaler),
                restart_fit=params['restart_fit'],
                validation_data=(x_val, y_val),
                callbacks=[EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)],
                epochs=200,
                verbose=0
            ),
            master=MTMaster(monotonicities[params['monotonicities']], alpha=params['alpha'], beta=params['beta']),
            init_step='pretraining',
            metrics=[AUC(name='auc')]
        )
        mt.fit(
            x=x_aug.values,
            y=y_aug['clicked'].values,
            iterations=10,
            val_data=dict(train=(x_train, y_train), val=(x_val, y_val), test=(x_test, y_test)),
            callbacks=[WandBLogger('shape_constraints', 'giuluck', 'restaurants', **params)]
        )
        print(f' -- elapsed time: {time.time() - start_time}')
    shutil.rmtree('wandb')
