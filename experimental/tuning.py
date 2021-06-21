"""Moving Target's Hyper-parameter Tuning & Investigation Script."""

import os

import wandb

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import shutil

from experimental.utils import DatasetFactory
from moving_targets.callbacks import WandBLogger
from src.util.dictionaries import cartesian_product

if __name__ == '__main__':
    # create study list
    study = cartesian_product(
        dataset=['cars univariate', 'cars', 'synthetic', 'puzzles'],
        mst_alpha=[0.01, 0.1, 1.0],
        mst_loss_fn=['mse'],
        mst_backend=['gurobi'],
        mst_learner_weights=['all'],
        mst_learner_omega=[1],
        mst_master_omega=[1],
        lrn_warm_start=[False]
    )

    # begin study
    for i, config in enumerate(study):
        start_time = time.time()
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end=' ')
        dataset = config.pop('dataset')
        try:
            if dataset == 'cars univariate':
                data_args = dict(full_features=False, full_grid=False)
            elif dataset == 'synthetic':
                data_args = dict(full_grid=False)
            else:
                data_args = dict(full_features=True, full_grid=False)

            factory, _ = DatasetFactory().get_dataset(name=dataset, data_args=data_args)
            factory.get_mt(**config).experiment(
                num_folds=10,
                iterations=50,
                fold_verbosity=True,
                callbacks=[WandBLogger(project='temp', entity='giuluck', run_name=dataset, **config, **data_args)]
            )
            print(f'-- elapsed time: {time.time() - start_time}')
        except:
            wandb.init(name=dataset, project='sc_classification', entity='giuluck', config={'crashed': True})
            wandb.finish()
            print('-- errors')
    shutil.rmtree('wandb')
