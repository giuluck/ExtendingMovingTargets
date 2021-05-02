import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import time

from moving_targets.callbacks import WandBLogger
from moving_targets.metrics import R2, MSE, MAE
from src.models import MT
from src.util.combinatorial import cartesian_product
from tests.regressions.models import Learner, Master
from tests.regressions.test import retrieve
from tests.util.experiments import setup

if __name__ == '__main__':
    study = cartesian_product(
        seed=[0, 1, 2, 3, 4],
        alpha=[0.01, 0.1, 1.0, 10.0],
        dataset=['cars_univariate'],
    ) + cartesian_product(
        seed=[0, 1, 2, 3, 4],
        alpha=[0.01, 0.1, 1.0, 10.0],
        master_omega=[1, 10, 100, 1000],
        learner_omega=[1, 10, 100, 1000],
        learner_y=['original', 'augmented'],
        learner_weights=['all', 'memory'],
        dataset=['cars', 'synthetic', 'puzzles'],
    )
    print(len(study))
    exit()

    # begin study
    for i, p in enumerate(study):
        setup(seed=p['seed'])
        start_time = time.time()
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}:', end='')
        dataset = p['dataset']
        config = {k: v for k, v in p.items() if k not in ['seed', 'dataset']}
        x_aug, y_aug, mono, data, aug_mk, _ = retrieve(dataset, 'group', aug=None, ground=None, supervised=False)
        mt = MT(
            learner=Learner(),
            master=Master(monotonicities=mono, augmented_mask=aug_mk, **config),
            init_step='pretraining',
            metrics=[MAE(), MSE(), R2()]
        )
        try:
            mt.fit(
                x=x_aug,
                y=y_aug,
                iterations=50,
                val_data={k: v for k, v in data.items() if k != 'scalers'},
                callbacks=[WandBLogger(project='sc', entity='giuluck', run_name=dataset, **config)],
                verbose=False
            )
            print(' -- elapsed time:', time.time() - start_time)
        except RuntimeError:
            print(' -- unsolvable')
            WandBLogger.instance.config.update({'crashed': True})
            WandBLogger.instance.finish()

    shutil.rmtree('wandb')
