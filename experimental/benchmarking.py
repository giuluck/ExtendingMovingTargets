"""Benchmarking Script."""

import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import shutil

from experimental.utils import DatasetFactory

if __name__ == '__main__':
    study = [DatasetFactory().get_dataset(name=dataset)[0].get_mt(
        mt_iterations=5,
        mst_master_kind='classification',
        lrn_loss='binary_crossentropy',
        mst_loss_fn='rbce',
        mst_alpha=alpha,
        wandb_name=f'MT BCE {alpha}'
    ) for alpha in [0.1, 1.0] for dataset in ['default', 'law']]

    # begin study
    for i, manager in enumerate(study):
        start_time = time.time()
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end=' ')
        manager.validate(num_folds=10, summary_args=None)
        print(f'-- elapsed time: {time.time() - start_time}')
    shutil.rmtree('wandb')
