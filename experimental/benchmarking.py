"""Benchmarking Script."""

import os

from experimental.utils import DatasetFactory

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import shutil

if __name__ == '__main__':
    alphas = [0.01, 0.1, 1.0]
    study = []
    for dataset in ['cars', 'synthetic', 'puzzles', 'restaurants', 'default', 'law']:
        factory, _ = DatasetFactory().get_dataset(name=dataset)
        benchmarks = [
            factory.get_mlp(wandb_name='MLP'),
            factory.get_sbr(wandb_name='SBR'),
            factory.get_tfl(wandb_name='TFL')
        ]
        if dataset in ['cars', 'synthetic', 'puzzles']:
            mts = [factory.get_mt(mt_iterations=20, mst_alpha=alpha, wandb_name=f'MT {alpha}') for alpha in alphas]
        else:
            mtr = [factory.get_mt(mt_iterations=20, mst_master_kind='regression', mst_loss_fn='mse',
                                  mst_alpha=alpha, wandb_name=f'MT MSE {alpha}') for alpha in alphas]
            mtc = [factory.get_mt(mt_iterations=5, mst_master_kind='classification', mst_loss_fn='rbce',
                                  mst_alpha=alpha, wandb_name=f'MT BCE {alpha}') for alpha in alphas]
            mts = mtr + mtc
        # noinspection PyTypeChecker
        study += benchmarks + mts

    # begin study
    for i, manager in enumerate(study):
        start_time = time.time()
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end=' ')
        try:
            manager.validate(num_folds=10, summary_args=None)
            print(f'-- elapsed time: {time.time() - start_time}')
        except:
            print('-- errors')
    shutil.rmtree('wandb')
