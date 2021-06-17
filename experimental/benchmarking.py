"""Benchmarking Script."""

import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import shutil
from typing import Dict

from experimental.utils import DatasetFactory
from experimental.utils.handlers import default_config

if __name__ == '__main__':
    # noinspection PyMissingOrEmptyDocstring
    def custom_config(handler, **kwargs) -> Dict:
        config = default_config(handler, **kwargs)
        config['dataset'] = config['dataset'] + ' full'
        return config


    cars, _ = DatasetFactory().cars(data_args=dict(full_features=True, full_grid=False))
    puzzles, _ = DatasetFactory().puzzles(data_args=dict(full_features=True, full_grid=False))
    law, _ = DatasetFactory().law(data_args=dict(full_features=True, full_grid=False))
    default, _ = DatasetFactory().default(data_args=dict(full_features=True, full_grid=False))

    study = [ds.get_mt(
        mt_iterations=20,
        mst_master_kind='regression',
        lrn_loss='mse',
        mst_loss_fn='mse',
        mst_alpha=alpha,
        wandb_name=f'MT MSE {alpha}',
        wandb_config=custom_config
    ) for alpha in [0.01, 0.1, 1.0] for ds in [cars, puzzles, law, default]]

    # begin study
    for i, manager in enumerate(study):
        start_time = time.time()
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end=' ')
        manager.validate(num_folds=10, summary_args=None)
        print(f'-- elapsed time: {time.time() - start_time}')
    shutil.rmtree('wandb')
