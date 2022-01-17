"""Benchmarking Script."""

import os
import warnings

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.exceptions import ConvergenceWarning

from src.managers import get_manager
from src.util.experiments import cartesian_product

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
        beta=[1.0],
        loss=['hd', 'ce', 'mse', 'mae'],
        adaptive=[False, True],
        dataset=['iris', 'redwine', 'whitewine', 'dota', 'shuttle', 'adult']
    ) + cartesian_product(
        init_step=['pretraining', 'projection'],
        alpha=[10.0, 1.0, 0.1, 0.01],
        beta=[1.0],
        loss=['mse', 'mae'],
        adaptive=[False, True],
        dataset=['communities']
    )

    for i, config in enumerate(study):
        manager = get_manager(**config)
        manager.get_wandb_logger(project='emt_parameters')
        start_time = time.time()
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end=' ')
        manager.experiment(
            iterations=iterations,
            num_folds=num_folds,
            model_verbosity=model_verbosity,
            fold_verbosity=fold_verbosity,
            callbacks=[manager.get_wandb_logger(project='emt_parameters')]
        )
        print(f'-- elapsed time: {time.time() - start_time}')

    shutil.rmtree('wandb')
