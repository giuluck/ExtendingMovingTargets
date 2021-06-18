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


    restaurants, _ = DatasetFactory().restaurants(data_args=dict(full_grid=True))
    default, _ = DatasetFactory().default(data_args=dict(full_features=False, full_grid=True, train_fraction=0.025))
    law, _ = DatasetFactory().law(data_args=dict(full_features=False, full_grid=True, train_fraction=0.03))

    study = [ds.get_mt(
        mt_iterations=5,
        mst_master_kind='classification',
        lrn_loss='binary_crossentropy',
        mst_backend='cvxpy',
        mst_loss_fn='rbce',
        mst_alpha=alpha,
        wandb_name=f'MT BCE {alpha} (CVX-SCS)',
        wandb_project='sc_benchmark',
        mst_custom_args=dict(solver='SCS')
    ) for ds in [restaurants, default, law] for alpha in [0.01, 0.1, 1.0]]

    # begin study
    for i, manager in enumerate(study):
        start_time = time.time()
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end=' ')
        manager.validate(num_folds=10, folds_index=[6, 7, 8, 9] if i == 0 else None, summary_args=None)
        print(f'-- elapsed time: {time.time() - start_time}')
    shutil.rmtree('wandb')
