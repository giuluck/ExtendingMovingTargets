"""Benchmarking Script."""

import os
import warnings

from sklearn.exceptions import ConvergenceWarning

from experimental.utils.configuration import get_manager
from moving_targets.callbacks import WandBLogger
from moving_targets.learners import LogisticRegression
from src.masters import BalancedCounts
from src.util.dictionaries import cartesian_product

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import shutil

from experimental.utils.handler import ExperimentHandler


def wandb_logger(h: ExperimentHandler) -> WandBLogger:
    return WandBLogger(project='emt_tuning', entity='giuluck', run_name=h.dataset, **h.wandb_config(h))


if __name__ == '__main__':
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    iterations = 15
    num_folds = None
    model_verbosity = False
    fold_verbosity = False

    study = cartesian_product(
        init_step=['pretraining', 'projection'],
        alpha=[1.0, 0.1, 0.01],
        beta=[1.0, 0.1, 0.01, None],
        loss_fn=['hd', 'ce', 'mse', 'mae'],
        dataset=['iris', 'redwine', 'whitewine', 'shuttle', 'dota']
    )

    for i, config in enumerate(study):
        manager = get_manager(dataset=config['dataset'])
        handler = ExperimentHandler(
            manager=manager,
            learner=LogisticRegression(),
            master=BalancedCounts(loss_fn=config['loss_fn'], alpha=config['alpha'], beta=config['beta']),
            init_step=config['init_step'],
            dataset=config['dataset']
        )
        start_time = time.time()
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end=' ')
        handler.experiment(num_folds=num_folds,
                           iterations=iterations,
                           model_verbosity=model_verbosity,
                           fold_verbosity=fold_verbosity,
                           callbacks=[wandb_logger(handler)])
        print(f'-- elapsed time: {time.time() - start_time}')

    shutil.rmtree('wandb')
