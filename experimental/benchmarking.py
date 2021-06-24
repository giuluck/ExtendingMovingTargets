"""Benchmarking Script."""

import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import shutil
from typing import Dict, List

from experimental.utils import DatasetFactory
from experimental.utils.handlers import default_config, AbstractHandler

if __name__ == '__main__':
    def _full_config(handler: AbstractHandler, **kwargs) -> Dict:
        config = default_config(handler, **kwargs)
        config['dataset'] = config['dataset'] + ' full'
        return config


    def _slim_config(handler: AbstractHandler, **kwargs) -> Dict:
        config = default_config(handler, **kwargs)
        config['dataset'] = config['dataset'] + ' slim'
        return config


    def _study(study: List[AbstractHandler]):
        print(study[0].wandb_config(study[0])['dataset'].upper())
        for i, manager in enumerate(study):
            start_time = time.time()
            print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end=' ')
            manager.validate(num_folds=5, summary_args=None)
            print(f'-- elapsed time: {time.time() - start_time}')
        print()


    WANDB_PROJECT: str = 'movingtarg_benchmarking'

    # BASELINES
    cars_full, _ = DatasetFactory().cars(data_args=dict(full_features=True, full_grid=False))
    cars_slim, _ = DatasetFactory().cars(data_args=dict(full_features=False, full_grid=False))
    cars_univ, _ = DatasetFactory().cars_univariate(data_args=dict(full_features=False, full_grid=False))
    _study([
        cars_full.get_mlp(wandb_name='MLP', wandb_project=WANDB_PROJECT, wandb_config=_full_config),
        cars_full.get_sbr(wandb_name='SBR', wandb_project=WANDB_PROJECT, wandb_config=_full_config),
        cars_full.get_tfl(wandb_name='TFL', wandb_project=WANDB_PROJECT, wandb_config=_full_config),
        cars_slim.get_mlp(wandb_name='MLP', wandb_project=WANDB_PROJECT, wandb_config=_slim_config),
        cars_slim.get_sbr(wandb_name='SBR', wandb_project=WANDB_PROJECT, wandb_config=_slim_config),
        cars_univ.get_univariate_sbr(wandb_name='SBR (No Augmentation)', wandb_project=WANDB_PROJECT,
                                     wandb_config=_slim_config),
        cars_slim.get_tfl(wandb_name='TFL', wandb_project=WANDB_PROJECT, wandb_config=_slim_config)
    ])

    synthetic, _ = DatasetFactory().synthetic(data_args=dict(full_grid=False))
    _study([
        synthetic.get_mlp(wandb_name='MLP', wandb_project=WANDB_PROJECT),
        synthetic.get_sbr(wandb_name='SBR', wandb_project=WANDB_PROJECT),
        synthetic.get_tfl(wandb_name='TFL', wandb_project=WANDB_PROJECT)
    ])

    puzzles_full, _ = DatasetFactory().puzzles(data_args=dict(full_features=True, full_grid=False))
    puzzles_slim, _ = DatasetFactory().puzzles(data_args=dict(full_features=False, full_grid=False))
    _study([
        puzzles_full.get_mlp(wandb_name='MLP', wandb_project=WANDB_PROJECT, wandb_config=_full_config),
        puzzles_full.get_sbr(wandb_name='SBR', wandb_project=WANDB_PROJECT, wandb_config=_full_config),
        puzzles_full.get_tfl(wandb_name='TFL', wandb_project=WANDB_PROJECT, wandb_config=_full_config),
        puzzles_slim.get_mlp(wandb_name='MLP', wandb_project=WANDB_PROJECT, wandb_config=_slim_config),
        puzzles_slim.get_sbr(wandb_name='SBR', wandb_project=WANDB_PROJECT, wandb_config=_slim_config),
        puzzles_slim.get_tfl(wandb_name='TFL', wandb_project=WANDB_PROJECT, wandb_config=_slim_config)
    ])

    restaurants, _ = DatasetFactory().restaurants(data_args=dict(full_grid=False))
    _study([
        restaurants.get_mlp(wandb_name='MLP', wandb_project=WANDB_PROJECT),
        restaurants.get_sbr(wandb_name='SBR', wandb_project=WANDB_PROJECT),
        restaurants.get_tfl(wandb_name='TFL', wandb_project=WANDB_PROJECT)
    ])

    default_full, _ = DatasetFactory().default(data_args=dict(full_features=True, full_grid=False))
    default_slim, _ = DatasetFactory().default(data_args=dict(full_features=False, full_grid=False))
    _study([
        default_full.get_mlp(wandb_name='MLP', wandb_project=WANDB_PROJECT, wandb_config=_full_config),
        default_full.get_sbr(wandb_name='SBR', wandb_project=WANDB_PROJECT, wandb_config=_full_config),
        default_slim.get_mlp(wandb_name='MLP', wandb_project=WANDB_PROJECT, wandb_config=_slim_config),
        default_slim.get_sbr(wandb_name='SBR', wandb_project=WANDB_PROJECT, wandb_config=_slim_config),
        default_slim.get_tfl(wandb_name='TFL', wandb_project=WANDB_PROJECT, wandb_config=_slim_config)
    ])

    law_full, _ = DatasetFactory().law(data_args=dict(full_features=True, full_grid=False))
    law_slim, _ = DatasetFactory().law(data_args=dict(full_features=False, full_grid=False))
    _study([
        law_full.get_mlp(wandb_name='MLP', wandb_project=WANDB_PROJECT, wandb_config=_full_config),
        law_full.get_sbr(wandb_name='SBR', wandb_project=WANDB_PROJECT, wandb_config=_full_config),
        law_slim.get_mlp(wandb_name='MLP', wandb_project=WANDB_PROJECT, wandb_config=_slim_config),
        law_slim.get_sbr(wandb_name='SBR', wandb_project=WANDB_PROJECT, wandb_config=_slim_config),
        law_slim.get_tfl(wandb_name='TFL', wandb_project=WANDB_PROJECT, wandb_config=_slim_config)
    ])

    # MOVING TARGETS
    _study([
        cars_full.get_mt(wandb_name='MT 0.01', wandb_project=WANDB_PROJECT, mt_iterations=30, mst_alpha=0.01,
                         wandb_config=_full_config),
        cars_full.get_mt(wandb_name='MT 0.1', wandb_project=WANDB_PROJECT, mt_iterations=30, mst_alpha=0.1,
                         wandb_config=_full_config),
        cars_full.get_mt(wandb_name='MT 1.0', wandb_project=WANDB_PROJECT, mt_iterations=30, mst_alpha=1.0,
                         wandb_config=_full_config),
        cars_slim.get_mt(wandb_name='MT 0.01', wandb_project=WANDB_PROJECT, mt_iterations=30, mst_alpha=0.01,
                         wandb_config=_slim_config),
        cars_slim.get_mt(wandb_name='MT 0.1', wandb_project=WANDB_PROJECT, mt_iterations=30, mst_alpha=0.1,
                         wandb_config=_slim_config),
        cars_slim.get_mt(wandb_name='MT 1.0', wandb_project=WANDB_PROJECT, mt_iterations=30, mst_alpha=1.0,
                         wandb_config=_slim_config),
        cars_univ.get_mt(wandb_name='MT 0.01 (No Augmentation)', wandb_project=WANDB_PROJECT, mt_iterations=30,
                         mst_alpha=0.01, wandb_config=_slim_config),
        cars_univ.get_mt(wandb_name='MT 0.1 (No Augmentation)', wandb_project=WANDB_PROJECT, mt_iterations=30,
                         mst_alpha=0.1, wandb_config=_slim_config),
        cars_univ.get_mt(wandb_name='MT 1.0 (No Augmentation)', wandb_project=WANDB_PROJECT, mt_iterations=30,
                         mst_alpha=1.0, wandb_config=_slim_config)
    ])

    _study([
        synthetic.get_mt(wandb_name='MT 0.01', wandb_project=WANDB_PROJECT, mt_iterations=30, mst_alpha=0.01),
        synthetic.get_mt(wandb_name='MT 0.1', wandb_project=WANDB_PROJECT, mt_iterations=30, mst_alpha=0.1),
        synthetic.get_mt(wandb_name='MT 1.0', wandb_project=WANDB_PROJECT, mt_iterations=30, mst_alpha=1.0)
    ])

    _study([
        puzzles_full.get_mt(wandb_name='MT 0.01', wandb_project=WANDB_PROJECT, mt_iterations=30, mst_alpha=0.01,
                            wandb_config=_full_config),
        puzzles_full.get_mt(wandb_name='MT 0.1', wandb_project=WANDB_PROJECT, mt_iterations=30, mst_alpha=0.1,
                            wandb_config=_full_config),
        puzzles_full.get_mt(wandb_name='MT 1.0', wandb_project=WANDB_PROJECT, mt_iterations=30, mst_alpha=1.0,
                            wandb_config=_full_config),
        puzzles_slim.get_mt(wandb_name='MT 0.01', wandb_project=WANDB_PROJECT, mt_iterations=30, mst_alpha=0.01,
                            wandb_config=_slim_config),
        puzzles_slim.get_mt(wandb_name='MT 0.1', wandb_project=WANDB_PROJECT, mt_iterations=30, mst_alpha=0.1,
                            wandb_config=_slim_config),
        puzzles_slim.get_mt(wandb_name='MT 1.0', wandb_project=WANDB_PROJECT, mt_iterations=30, mst_alpha=1.0,
                            wandb_config=_slim_config)

    ])

    _study([
        restaurants.get_mt(wandb_name='MT 0.01', wandb_project=WANDB_PROJECT, mt_iterations=10, mst_alpha=0.01,
                           mst_master_kind='regression', mst_loss_fn='mse', lrn_loss='mse'),
        restaurants.get_mt(wandb_name='MT 0.1', wandb_project=WANDB_PROJECT, mt_iterations=10, mst_alpha=0.1,
                           mst_master_kind='regression', mst_loss_fn='mse', lrn_loss='mse'),
        restaurants.get_mt(wandb_name='MT 1.0', wandb_project=WANDB_PROJECT, mt_iterations=10, mst_alpha=1.0,
                           mst_master_kind='regression', mst_loss_fn='mse', lrn_loss='mse')
        # rest.get_mt(wandb_name='MT BCE 0.01', wandb_project=WANDB_PROJECT, mt_iterations=5, mst_alpha=0.01,
        #             mst_master_kind='classification', mst_loss_fn='binary_crossentropy', lrn_loss='rbce'),
        # rest.get_mt(wandb_name='MT BCE 0.1', wandb_project=WANDB_PROJECT, mt_iterations=5, mst_alpha=0.1,
        #             mst_master_kind='classification', mst_loss_fn='binary_crossentropy', lrn_loss='rbce'),
        # rest.get_mt(wandb_name='MT BCE 1.0', wandb_project=WANDB_PROJECT, mt_iterations=5, mst_alpha=1.0,
        #             mst_master_kind='classification', mst_loss_fn='binary_crossentropy', lrn_loss='rbce')
    ])

    _study([
        default_full.get_mt(wandb_name='MT 0.01', wandb_project=WANDB_PROJECT, mt_iterations=10, mst_alpha=0.01,
                            mst_master_kind='regression', mst_loss_fn='mse', lrn_loss='mse', wandb_config=_full_config),
        default_full.get_mt(wandb_name='MT 0.1', wandb_project=WANDB_PROJECT, mt_iterations=10, mst_alpha=0.1,
                            mst_master_kind='regression', mst_loss_fn='mse', lrn_loss='mse', wandb_config=_full_config),
        default_full.get_mt(wandb_name='MT 1.0', wandb_project=WANDB_PROJECT, mt_iterations=10, mst_alpha=1.0,
                            mst_master_kind='regression', mst_loss_fn='mse', lrn_loss='mse', wandb_config=_full_config),
        default_slim.get_mt(wandb_name='MT 0.01', wandb_project=WANDB_PROJECT, mt_iterations=10, mst_alpha=0.01,
                            mst_master_kind='regression', mst_loss_fn='mse', lrn_loss='mse', wandb_config=_slim_config),
        default_slim.get_mt(wandb_name='MT 0.1', wandb_project=WANDB_PROJECT, mt_iterations=10, mst_alpha=0.1,
                            mst_master_kind='regression', mst_loss_fn='mse', lrn_loss='mse', wandb_config=_slim_config),
        default_slim.get_mt(wandb_name='MT 1.0', wandb_project=WANDB_PROJECT, mt_iterations=10, mst_alpha=1.0,
                            mst_master_kind='regression', mst_loss_fn='mse', lrn_loss='mse', wandb_config=_slim_config),
    ])

    _study([
        law_full.get_mt(wandb_name='MT 0.01', wandb_project=WANDB_PROJECT, mt_iterations=10, mst_alpha=0.01,
                        mst_master_kind='regression', mst_loss_fn='mse', lrn_loss='mse', wandb_config=_full_config),
        law_full.get_mt(wandb_name='MT 0.1', wandb_project=WANDB_PROJECT, mt_iterations=10, mst_alpha=0.1,
                        mst_master_kind='regression', mst_loss_fn='mse', lrn_loss='mse', wandb_config=_full_config),
        law_full.get_mt(wandb_name='MT 1.0', wandb_project=WANDB_PROJECT, mt_iterations=10, mst_alpha=1.0,
                        mst_master_kind='regression', mst_loss_fn='mse', lrn_loss='mse', wandb_config=_full_config),
        law_slim.get_mt(wandb_name='MT 0.01', wandb_project=WANDB_PROJECT, mt_iterations=10, mst_alpha=0.01,
                        mst_master_kind='regression', mst_loss_fn='mse', lrn_loss='mse', wandb_config=_slim_config),
        law_slim.get_mt(wandb_name='MT 0.1', wandb_project=WANDB_PROJECT, mt_iterations=10, mst_alpha=0.1,
                        mst_master_kind='regression', mst_loss_fn='mse', lrn_loss='mse', wandb_config=_slim_config),
        law_slim.get_mt(wandb_name='MT 1.0', wandb_project=WANDB_PROJECT, mt_iterations=10, mst_alpha=1.0,
                        mst_master_kind='regression', mst_loss_fn='mse', lrn_loss='mse', wandb_config=_slim_config)
    ])

    shutil.rmtree('wandb')
