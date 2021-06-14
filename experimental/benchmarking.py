"""Benchmarking Script."""

import os
from typing import Dict

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import shutil

from experimental.utils import DatasetFactory
from experimental.utils.handlers import default_config

if __name__ == '__main__':
    # noinspection PyMissingOrEmptyDocstring
    def custom_config(handler, **kwargs) -> Dict:
        config = default_config(handler)
        config['dataset'] = config['dataset'] + ' 0.8'
        return config


    law, _ = DatasetFactory().law(data_args=dict(full_features=False, full_grid=True, train_fraction=0.8))
    default, _ = DatasetFactory().default(data_args=dict(full_features=False, full_grid=True, train_fraction=0.8))
    study = []
    for ds in [law, default]:
        study += [
            ds.get_mlp(wandb_name='MLP', wandb_config=custom_config),
            ds.get_sbr(wandb_name='SBR', wandb_config=custom_config),
            ds.get_tfl(wandb_name='TFL', wandb_config=custom_config)
        ]
        study += [ds.get_mt(
            mt_iterations=20,
            mst_master_kind='regression',
            lrn_loss='mse',
            mst_loss_fn='mse',
            mst_alpha=alpha,
            wandb_name=f'MT MSE {alpha}',
            wandb_config=custom_config
        ) for alpha in [0.01, 0.1, 1.0]]

    # begin study
    for i, manager in enumerate(study):
        start_time = time.time()
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end=' ')
        manager.validate(num_folds=10, summary_args=None)
        print(f'-- elapsed time: {time.time() - start_time}')
    shutil.rmtree('wandb')
