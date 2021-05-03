import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import shutil
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from src import restaurants
from moving_targets.callbacks import WandBLogger
from moving_targets.metrics import AUC, MAE, MSE, MonotonicViolation
from src.models import MT, MTLearner, MTClassificationMaster
from src.restaurants import compute_monotonicities
from src.restaurants.augmentation import get_augmented_data
from src.util.augmentation import get_monotonicities_list
from src.util.combinatorial import cartesian_product
from tests.restaurants.test import neural_model
from tests.util.experiments import setup


# noinspection PyUnusedLocal
def on_training_end(model, macs, x, y, val_data, iteration):
    model.log(**{'metrics/ground_r2': model.compute_ground_r2()})


if __name__ == '__main__':
    # import extension methods and ground_r2 log method
    restaurants.import_extension_methods()
    MT.on_training_end = on_training_end

    # load and prepare data
    data = restaurants.load_data()
    x_aug, y_aug, full_aug, aug_scaler = get_augmented_data(data['train'][0], data['train'][1])
    aug_mask = np.isnan(y_aug['clicked'])
    mono = get_monotonicities_list(full_aug, compute_monotonicities, 'clicked', 'group')

    # create study list
    study = cartesian_product(
        seed=[0, 1, 2],
        alpha=[0.01, 0.1, 1.0],
        master_omega=[1, 10, 100],
        learner_omega=[1, 10, 100],
        learner_y=['original', 'augmented', 'adjusted'],
        learner_weights=['all', 'memory']
    )

    # begin study
    for i, p in enumerate(study):
        start_time = time.time()
        setup(seed=p['seed'])
        config = {k: v for k, v in p.items() if k not in ['seed']}
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end='')
        es = EarlyStopping(monitor='loss', patience=10, min_delta=1e-4)
        mt = MT(
            learner=MTLearner(lambda: neural_model([16, 8, 8], aug_scaler), epochs=200, callbacks=[es], verbose=False),
            master=MTClassificationMaster(monotonicities=mono, augmented_mask=aug_mask, **config),
            init_step='pretraining',
            metrics=[MAE(), MSE(), AUC(),
                     MonotonicViolation(monotonicities=mono, aggregation='average', name='avg. violation'),
                     MonotonicViolation(monotonicities=mono, aggregation='percentage', name='pct. violation'),
                     MonotonicViolation(monotonicities=mono, aggregation='feasible', name='is feasible')]
        )
        try:
            mt.fit(
                x=x_aug,
                y=y_aug,
                iterations=10,
                val_data={k: v for k, v in data.items() if k != 'scalers'},
                callbacks=[WandBLogger(project='sc', entity='giuluck', run_name='restaurants', **config)],
                verbose=False
            )
            print(' -- elapsed time:', time.time() - start_time)
        except RuntimeError:
            print(' -- unsolvable')
            WandBLogger.instance.config.update({'crashed': True})
            WandBLogger.instance.finish()
        print(f' -- elapsed time: {time.time() - start_time}')
    shutil.rmtree('wandb')
