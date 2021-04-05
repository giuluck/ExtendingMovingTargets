import os

from src.util.augmentation import get_monotonicities_list

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src import restaurants
from moving_targets.callbacks import WandBLogger
from moving_targets.metrics import AUC
from src.models import MLP, MT, MTLearner, MTMaster
from src.restaurants import compute_monotonicities
from src.restaurants.augmentation import get_augmented_data
from src.util.combinatorial import cartesian_product


def get_model(h_units, scaler):
    model = MLP(output_act='sigmoid', h_units=h_units, scaler=scaler)
    model.compile(optimizer='adam', loss='mse')
    return model


def on_training_end(model, macs, x, y, val_data, iteration):
    model.log(**{'learner/ground_r2': model.compute_ground_r2()})


if __name__ == '__main__':
    # set random seeds and import extension methods
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)
    restaurants.import_extension_methods()
    MT.on_training_end = on_training_end

    # load and prepare data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = restaurants.load_data()
    x_aug, y_aug, full_aug, aug_scaler = get_augmented_data(x_train, y_train)
    monotonicities = get_monotonicities_list(full_aug, compute_monotonicities, 'clicked', ['ground', 'group', 'all'])

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
                epochs=0,
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
