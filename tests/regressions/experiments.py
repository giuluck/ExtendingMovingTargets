import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import time
import numpy as np

from moving_targets.callbacks import WandBLogger
from moving_targets.metrics import R2, MSE, MAE
from src.models import MT
from src.util.combinatorial import cartesian_product
from tests.regressions.models import Learner, Master
from tests.regressions.test import retrieve
from tests.util.experiments import setup

if __name__ == '__main__':
    setup()

    master_args = cartesian_product(
        beta_method=['none'],
        beta=[1.0],
        perturbation_method=['none'],
        perturbation=[None],
        loss_fn=['mae', 'mse'],
        alpha=[0.1, 0.5, 1.0, 2.0, 10.0],
        gamma=[1, 7.5, 15],
        master_weights=[1, 7.5, 15],
        weight_method=['gamma', 'memory-prop', 'memory-step', 'memory-same', 'memory-inc']
    )
    # master_args = [{k: v for d in ma.values() for k, v in d.items()} for ma in master_args]

    study = cartesian_product(
        dataset=['puzzles'],
        mono=['ground'],
        learner_args=[dict(backend='keras', optimizer='adam', warm_start=False)],
        master_args=master_args
    )

    # begin study
    for i, p in enumerate(study):
        start_time = time.time()
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end='')
        x_aug, y_aug, mono, data, aug_mk, _ = retrieve(p['dataset'], p['mono'], aug=None, ground=None, supervised=False)
        mt = MT(
            learner=Learner(**p['learner_args']),
            master=Master(monotonicities=mono, augmented_mask=aug_mk, **p['master_args']),
            init_step='pretraining',
            metrics=[MAE(), MSE(), R2()]
        )
        try:
            config = {'monotonicity': p['mono'], **p['learner_args'], **p['master_args']}
            mt.fit(
                x=x_aug,
                y=y_aug,
                iterations=10,
                sample_weight=np.where(aug_mk, 1 / 15, 1),
                val_data={k: v for k, v in data.items() if k != 'scalers'},
                callbacks=[WandBLogger(project='sc', entity='giuluck', run_name=p['dataset'], **config)],
                verbose=False
            )
            print(' -- elapsed time:', time.time() - start_time)
        except RuntimeError:
            print(' -- unsolvable')
            WandBLogger.instance.config.update({'crashed': True})
            WandBLogger.instance.finish()

    shutil.rmtree('wandb')
