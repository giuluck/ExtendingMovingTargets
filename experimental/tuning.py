"""Moving Target's Hyper-parameter Tuning & Investigation Script."""

import os

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
        mst_loss_fn=['mae', 'mse/bce', 'mae/bce'],
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
        factory, _ = DatasetFactory().get_dataset(name=dataset)
        # handle losses configuration
        mt_config = config.copy()
        mt_config['mst_master_kind'] = mt_config.pop('master_kind')
        mst_loss_fn = mt_config.pop('mst_loss_fn')
        if mst_loss_fn == 'mae':
            mt_config['lrn_loss'] = 'mse'
            mt_config['mst_loss_fn'] = 'mae'
        elif mst_loss_fn == 'mse/bce':
            mt_config['lrn_loss'] = 'binary crossentropy'
            mt_config['mst_loss_fn'] = 'mse'
        elif mst_loss_fn == 'mae/bce':
            mt_config['lrn_loss'] = 'binary crossentropy'
            mt_config['mst_loss_fn'] = 'mae'
        else:
            raise ValueError(f"unknown mst_loss_fn '{mst_loss_fn}'")
        # run cross-validation
        factory.get_mt(**mt_config).experiment(
            num_folds=5,
            iterations=10,
            fold_verbosity=True,
            callbacks=[WandBLogger(project='sc_classification', entity='giuluck', run_name=dataset, **config)]
        )
        print(f'-- elapsed time: {time.time() - start_time}')
    shutil.rmtree('wandb')
