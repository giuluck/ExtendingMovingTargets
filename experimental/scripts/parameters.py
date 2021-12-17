"""Benchmarking Script."""

import os
import warnings

from sklearn.exceptions import ConvergenceWarning

from experimental.utils.configuration import get_manager
from moving_targets.callbacks import WandBLogger
from moving_targets.learners import *
from src.masters import *
from src.util.dictionaries import cartesian_product, merge_dictionaries

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import shutil

from experimental.utils.handler import ExperimentHandler


def wandb_logger(h: ExperimentHandler, **c) -> WandBLogger:
    c = merge_dictionaries(h.wandb_config(h), c)
    return WandBLogger(project='emt_parameters', entity='giuluck', run_name=h.dataset, **c)


if __name__ == '__main__':
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    iterations = 15
    num_folds = 5
    model_verbosity = False
    fold_verbosity = False

    study = cartesian_product(
        init_step=['pretraining', 'projection'],
        alpha=[10.0, 1.0, 0.1, 0.01],
        beta=[1.0, 0.1, 0.01, None],
        loss_fn=['hd', 'ce', 'mse', 'mae'],
        dataset=['iris', 'redwine', 'whitewine', 'dota', 'shuttle', 'adult']
    ) + cartesian_product(
        init_step=['pretraining', 'projection'],
        alpha=[10.0, 1.0, 0.1, 0.01],
        beta=[1.0, 0.1, 0.01, None],
        loss_fn=['mse', 'mae'],
        dataset=['communities']
    )

    for i, cfg in enumerate(study):
        if cfg['dataset'] == 'communities':
            manager = get_manager('communities')
            learner = LinearRegression()
            master = FairRegression(protected='race', loss_fn=cfg['loss_fn'], alpha=cfg['alpha'], beta=cfg['beta'])
        elif cfg['dataset'] == 'adult':
            manager = get_manager('adult')
            learner = LogisticRegression()
            master = FairClassification(protected='race', loss_fn=cfg['loss_fn'], alpha=cfg['alpha'], beta=cfg['beta'])
        else:
            manager = get_manager(cfg['dataset'])
            learner = LogisticRegression()
            master = BalancedCounts(loss_fn=cfg['loss_fn'], alpha=cfg['alpha'], beta=cfg['beta'])
        handler = ExperimentHandler(
            manager=manager,
            learner=learner,
            master=master,
            init_step=cfg['init_step'],
            iterations=iterations,
            dataset=cfg['dataset']
        )
        start_time = time.time()
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end=' ')
        handler.experiment(
            num_folds=num_folds,
            model_verbosity=model_verbosity,
            fold_verbosity=fold_verbosity,
            callbacks=[wandb_logger(handler, loss=cfg['loss_fn'])]
        )
        print(f'-- elapsed time: {time.time() - start_time}')

    shutil.rmtree('wandb')
