import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import time
from tensorflow.python.keras.callbacks import EarlyStopping

from moving_targets.callbacks import WandBLogger
from moving_targets.metrics import R2, MSE, MAE, MonotonicViolation
from src.models import MT, MTRegressionMaster, MTLearner
from src.util.combinatorial import cartesian_product
from tests.regressions.test import retrieve
from tests.util.experiments import setup

if __name__ == '__main__':
    study = cartesian_product(
        seed=[0, 1, 2],
        alpha=[0.01, 0.1, 1.0],
        master_omega=[1, 10, 100],
        learner_omega=[1, 10, 100],
        learner_y=['original', 'augmented', 'adjusted'],
        learner_weights=['all', 'infeasible'],
        dataset=['puzzles'],
    )

    # begin study
    for i, p in enumerate(study):
        start_time = time.time()
        setup(seed=p['seed'])
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}:', end='')
        x_aug, y_aug, mono, data, aug_mk, _ = retrieve(p['dataset'], 'group', aug=None, ground=None, supervised=False)
        master_args = {k: v for k, v in p.items() if k not in ['seed', 'dataset']}
        es = EarlyStopping(monitor='loss', patience=10, min_delta=1e-4)
        mt = MT(
            learner=MTLearner(output_act=None, h_units=[16] * 4, optimizer='adam', loss='mse', warm_start=False,
                              epochs=200, callbacks=[es], verbose=False),
            master=MTRegressionMaster(monotonicities=mono, augmented_mask=aug_mk, **master_args),
            init_step='pretraining',
            metrics=[MAE(), MSE(), R2(),
                     MonotonicViolation(monotonicities=mono, aggregation='average', name='avg. violation'),
                     MonotonicViolation(monotonicities=mono, aggregation='percentage', name='pct. violation'),
                     MonotonicViolation(monotonicities=mono, aggregation='feasible', name='is feasible')]
        )
        try:
            mt.fit(
                x=x_aug,
                y=y_aug,
                iterations=50,
                val_data={k: v for k, v in data.items() if k != 'scalers'},
                callbacks=[WandBLogger(project='sc', entity='giuluck', run_name=p['dataset'], **p)],
                verbose=False
            )
            print(' -- elapsed time:', time.time() - start_time)
        except RuntimeError:
            print(' -- unsolvable')
            WandBLogger.instance.config.update({'crashed': True})
            WandBLogger.instance.finish()

    shutil.rmtree('wandb')
