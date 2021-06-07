"""Experiments Script."""

import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import shutil

from moving_targets.callbacks import WandBLogger
from src.util.dictionaries import cartesian_product
from experimental.datasets.tests import get_dataset

if __name__ == '__main__':
    # create study list
    study = cartesian_product(
        mst_backend=['gurobi'],
        mst_alpha=[0.01, 0.1, 1.0],
        mst_loss_fn=['mse', 'mae'],
        mst_learner_weights=['all', 'infeasible'],
        mst_learner_omega=[1, 10, 100],
        mst_master_omega=[1, 10, 100],
        lrn_warm_start=[True, False],
        dataset=['cars', 'synthetic', 'puzzles']
    )

    # begin study
    for i, config in enumerate(study):
        start_time = time.time()
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end=' ')
        dataset = config['dataset']
        del config['dataset']
        try:
            manager, _ = get_dataset(dataset=dataset)
            manager(**config).validate(
                num_folds=5,
                iterations=10,
                verbose=True,
                callbacks=[WandBLogger(project='sc_regression', entity='giuluck', run_name=dataset, **config)]
            )
            print(f'-- elapsed time: {time.time() - start_time}')
        except:
            print('-- errors')
    shutil.rmtree('wandb')
