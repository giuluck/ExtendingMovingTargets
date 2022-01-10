"""Benchmarking Script."""

import os
import warnings

from moving_targets.masters.backends import GurobiBackend

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from moving_targets.callbacks import WandBLogger
from sklearn.exceptions import ConvergenceWarning

from src.managers import get_manager
from src.util.dictionaries import cartesian_product

import time
import shutil

if __name__ == '__main__':
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    iterations = 15
    num_folds = None
    model_verbosity = False
    fold_verbosity = False

    study = cartesian_product(
        init_step=['pretraining', 'projection'],
        alpha=[10.0, 1.0, 0.1, 0.01],
        beta=[1.0, 0.1, 0.01, None],
        loss=['hd', 'ce', 'mse', 'mae'],
        adaptive=[False, True],
        dataset=['iris', 'redwine', 'whitewine', 'dota', 'shuttle', 'adult'],
        fixed_parameters=dict(backend=GurobiBackend(time_limit=30))
    ) + cartesian_product(
        init_step=['pretraining', 'projection'],
        alpha=[10.0, 1.0, 0.1, 0.01],
        beta=[1.0, 0.1, 0.01, None],
        loss=['mse', 'mae'],
        adaptive=[False, True],
        fixed_parameters=dict(backend=GurobiBackend(time_limit=30), dataset='communities')
    )

    for i, config in enumerate(study):
        manager = get_manager(**config)
        logger = WandBLogger(project='emt_parameters', entity='giuluck', run_name=manager.name(), **manager.kwargs)
        start_time = time.time()
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end=' ')
        manager.experiment(
            iterations=iterations,
            num_folds=num_folds,
            model_verbosity=model_verbosity,
            fold_verbosity=fold_verbosity,
            callbacks=[logger]
        )
        print(f'-- elapsed time: {time.time() - start_time}')

    shutil.rmtree('wandb')
