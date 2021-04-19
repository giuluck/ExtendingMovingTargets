import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import time

from moving_targets.callbacks import WandBLogger
from moving_targets.metrics import R2, MSE, MAE
from src.models import MT
from src.util.combinatorial import cartesian_product
from tests.regressions.models import Learner, UnsupervisedMaster
from tests.regressions.test import retrieve
from tests.util.experiments import setup

if __name__ == '__main__':
    setup()

    master_args = cartesian_product(
        general_args=[
            dict(loss_fn='mae', alpha=1.0)
        ],
        beta_args=[
            dict(beta_method='standard', beta=1.0),
            dict(beta_method='standard', beta=0.1),
            dict(beta_method='standard', beta=0.01),
            dict(beta_method='proportional', beta=1.0),
            dict(beta_method='none', beta=1.0)
        ],
        perturbation_args=[
            dict(perturbation_method='none', perturbation=None),
            dict(perturbation_method='loss', perturbation=0.01),
            dict(perturbation_method='loss', perturbation=0.1),
            dict(perturbation_method='constraint', perturbation=0.01),
            dict(perturbation_method='constraint', perturbation=0.1),
        ],
        weights_args=[
            dict(weight_method='uniform', gamma=None, min_weight=None),
            dict(weight_method='gamma', gamma=15, min_weight=None),
            dict(weight_method='distance', gamma=15, min_weight=0.0),
            dict(weight_method='distance', gamma=15, min_weight='gamma'),
            dict(weight_method='feasibility-prop', gamma=15, min_weight=0.0),
            dict(weight_method='feasibility-prop', gamma=15, min_weight='gamma'),
            dict(weight_method='feasibility-step', gamma=15, min_weight=0.0),
            dict(weight_method='feasibility-step', gamma=15, min_weight='gamma')
        ])
    master_args = [{k: v for d in ma.values() for k, v in d.items()} for ma in master_args]

    study = cartesian_product(
        dataset=['cars', 'synthetic'],
        monotonicity=['ground', 'group'],
        learner_args=[dict(backend='keras', optimizer='adam', warm_start=False)],
        master_args=master_args
    )

    # begin study
    for i, p in enumerate(study):
        start_time = time.time()
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end='')
        x_aug, y_aug, mono, data, _ = retrieve(p['dataset'], p['monotonicity'], aug=None, ground=None)
        mt = MT(
            learner=Learner(**p['learner_args']),
            master=UnsupervisedMaster(monotonicities=mono, **p['master_args']),
            init_step='pretraining',
            metrics=[MAE(), MSE(), R2()]
        )
        try:
            config = {'monotonicity': p['monotonicity'], **p['learner_args'], **p['master_args']}
            mt.fit(
                x=x_aug,
                y=y_aug,
                iterations=10,
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
