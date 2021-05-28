"""Experiments Script."""

import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import time

from experimental.datasets.managers import CarsTest, SyntheticTest, PuzzlesTest
from experimental.models.managers import MLPManager, SBRManager, UnivariateSBRManager, MTManager
from src.models import hard_tanh

if __name__ == '__main__':
    study = [
        # CARS
        MLPManager(test_manager=CarsTest(), wandb_name='MLP'),
        SBRManager(test_manager=CarsTest(), regularizer_act=None, wandb_name='SBR None'),
        UnivariateSBRManager(test_manager=CarsTest(), regularizer_act=None, direction=-1, wandb_name='SBR Uni'),
        MTManager(test_manager=CarsTest(mst_alpha=0.01), wandb_name='MT 0.01'),
        MTManager(test_manager=CarsTest(mst_alpha=0.1), wandb_name='MT 0.1'),
        MTManager(test_manager=CarsTest(mst_alpha=1.0), wandb_name='MT 1.0'),
        # SYNTHETIC
        MLPManager(test_manager=SyntheticTest(), wandb_name='MLP'),
        SBRManager(test_manager=SyntheticTest(), regularizer_act=None, wandb_name='SBR None'),
        SBRManager(test_manager=SyntheticTest(), regularizer_act=hard_tanh, wandb_name='SBR Tanh'),
        MTManager(test_manager=SyntheticTest(mst_alpha=0.01), wandb_name='MT 0.01'),
        MTManager(test_manager=SyntheticTest(mst_alpha=0.1), wandb_name='MT 0.1'),
        MTManager(test_manager=SyntheticTest(mst_alpha=1.0), wandb_name='MT 1.0'),
        # PUZZLES
        MLPManager(test_manager=PuzzlesTest(), wandb_name='MLP'),
        SBRManager(test_manager=PuzzlesTest(), regularizer_act=None, wandb_name='SBR None'),
        SBRManager(test_manager=PuzzlesTest(), regularizer_act=hard_tanh, wandb_name='SBR Tanh'),
        MTManager(test_manager=PuzzlesTest(mst_alpha=0.01), wandb_name='MT 0.01'),
        MTManager(test_manager=PuzzlesTest(mst_alpha=0.1), wandb_name='MT 0.1'),
        MTManager(test_manager=PuzzlesTest(mst_alpha=1.0), wandb_name='MT 1.0')
    ]

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
