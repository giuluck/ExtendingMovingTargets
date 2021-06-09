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
        mst_alpha=[0.01, 0.1, 1.0],
        dataset=['restaurants', 'default', 'law'],
        mst_loss_fn=['mse', 'mae'],
        mst_backend=['gurobi'],
        master_kind=['regression'],
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
            factory, _ = DatasetFactory().get_dataset(name=dataset)
            factory.get_mt(**config).experiment(
                num_folds=5,
                iterations=10,
                fold_verbosity=True,
                callbacks=[WandBLogger(project='sc_classification', entity='giuluck', run_name=dataset, **config)]
            )
            print(f'-- elapsed time: {time.time() - start_time}')
        except:
            wandb.init(name=dataset, project='sc_classification', entity='giuluck', config={'crashed': True})
            wandb.finish()
            print('-- errors')
    shutil.rmtree('wandb')
