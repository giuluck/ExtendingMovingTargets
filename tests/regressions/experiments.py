import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import time

from moving_targets.callbacks import WandBLogger
from moving_targets.metrics import R2, MSE, MAE
from src.models import MT
from src.util.combinatorial import cartesian_product
from tests.regressions.models import DistanceProportional, GammaWeighted, FeasibilityProportional, FeasibilityGamma, \
    Keras, Scikit, Uniform
from tests.regressions.test import retrieve
from tests.util.experiments import setup


def get_learner(name, **kwargs):
    if name == 'keras':
        return Keras(**kwargs)
    elif name == 'scikit':
        return Scikit(**kwargs)
    else:
        ValueError('Unknown learner name')


def get_master(name, **kwargs):
    if name == 'uniform':
        return Uniform(**kwargs)
    elif name == 'distance_proportional':
        return DistanceProportional(**kwargs)
    elif name == 'gamma_weighted':
        return GammaWeighted(**kwargs)
    elif name == 'feasibility_proportional':
        return FeasibilityProportional(**kwargs)
    elif name == 'feasibility_gamma':
        return FeasibilityGamma(**kwargs)
    else:
        ValueError('Unknown master name')


if __name__ == '__main__':
    setup()

    study = cartesian_product(
        dataset=['cars', 'synthetic', 'puzzles'],
        monotonicity=['group'],
        learner=['keras'],
        master=['uniform', 'distance_proportional', 'gamma_weighted', 'feasibility_proportional', 'feasibility_gamma'],
        warm_start=[True, False],
        prop_beta=[False],
        loss=['mse', 'mae', 'sse', 'sae']
    )

    # begin study
    for i, p in enumerate(study):
        start_time = time.time()
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end='')
        x_aug, y_aug, mono, data, _ = retrieve(p['dataset'], p['monotonicity'], aug=None, ground=None)
        mt = MT(
            learner=get_learner(p['learner'], warm_start=p['warm_start']),
            master=get_master(p['master'], monotonicities=mono, gamma=15, prop_beta=p['prop_beta'], loss_fn=p['loss']),
            init_step='pretraining',
            metrics=[MAE(), MSE(), R2()]
        )
        mt.fit(
            x=x_aug,
            y=y_aug,
            iterations=10,
            val_data={k: v for k, v in data.items() if k != 'scalers'},
            callbacks=[WandBLogger('shape_constraints_2', 'giuluck', 'trial', **p)],
            verbose=False
        )
        print(f' -- elapsed time: {time.time() - start_time}')
    shutil.rmtree('wandb')
