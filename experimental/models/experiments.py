"""Experiments Script."""

import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import time

from experimental.datasets.managers import CarsTest, SyntheticTest, PuzzlesTest, RestaurantsTest, DefaultTest, LawTest
from experimental.models.managers import MLPManager, SBRManager, MTManager, TFLManager

if __name__ == '__main__':
    alphas = [0.01, 0.1, 1.0]
    for manager in [CarsTest, SyntheticTest, PuzzlesTest, RestaurantsTest, DefaultTest, LawTest]:
        benchmarks = [
            MLPManager(test_manager=manager(), wandb_name='MLP'),
            SBRManager(test_manager=manager(), wandb_name='SBR'),
            TFLManager(test_manager=manager(), wandb_name='TFL')
        ]
        if manager in [CarsTest, SyntheticTest, PuzzlesTest]:
            mts = [MTManager(test_manager=manager(mst_alpha=a), iterations=20, wandb_name=f'MT {a}') for a in alphas]
        else:
            mtc = [MTManager(test_manager=manager(mst_alpha=a, master_kind='classification', mst_loss_fn='rbce'),
                             iterations=10, wandb_name=f'MT BCE {a}') for a in alphas]
            mtr = [MTManager(test_manager=manager(mst_alpha=a, master_kind='regression', mst_loss_fn='mse'),
                             iterations=20, wandb_name=f'MT MSE {a}') for a in alphas]
            mts = mtc + mtr
    # noinspection PyTypeChecker
    study = benchmarks + mts
    for i, s in enumerate(study):
        print(i, '-->', s)
    exit()

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
