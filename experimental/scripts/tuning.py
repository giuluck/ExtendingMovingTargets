"""Moving Targets Hyper-parameter Tuning & Investigation Script."""

import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import shutil
from typing import List, Dict

from experimental.utils.factories import DatasetFactory
from moving_targets.callbacks import WandBLogger
from src.util.dictionaries import cartesian_product

if __name__ == '__main__':
    def _study(study: List[Dict], dataset: str, name: str, data_args: Dict = None):
        print(name.upper())
        factory, _ = DatasetFactory().get_dataset(name=dataset, data_args=data_args)
        for i, config in enumerate(study):
            start_time = time.time()
            print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end=' ')
            mt = factory.get_mt(wandb_name=None, **config)
            mt.experiment(
                num_folds=5,
                iterations=30,
                fold_verbosity=False,
                callbacks=[WandBLogger(project=WANDB_PROJECT, entity='giuluck', run_name=name, **mt.wandb_config(mt))]
            )
            print(f'-- elapsed time: {time.time() - start_time}')
        print()


    WANDB_PROJECT: str = 'mt_tuning'

    # PRELIMINARY STUDIES
    preliminary = cartesian_product(
        fixed_parameters=dict(
            mst_backend='gurobi',
            mst_learner_weights='all',
            mst_learner_omega=1,
            mst_master_omega=1,
            mst_master_kind='regression',
            lrn_warm_start=False,
            lrn_loss='mse'
        ),
        mst_alpha=[0.01, 0.1, 1.0],
        mst_beta=[0.01, 0.1, 1.0, None],
        mst_loss_fn=['mae', 'mse'],
        mnt_kind=['ground', 'group', 'all'],
        mt_init_step=['pretraining', 'projection']
    )
    _study(preliminary, dataset='synthetic', name='preliminary', data_args=dict(full_grid=False))

    # STUDIES ON REGRESSION DATASET
    regression = cartesian_product(
        fixed_parameters=dict(
            mst_backend='gurobi',
            mst_beta=None,
            mnt_kind='group',
            mt_init_step='pretraining',
            mst_master_kind='regression',
            lrn_loss='mse'
        ),
        mst_alpha=[0.01, 0.1, 1.0],
        mst_learner_omega=[1.0, 10.0, 100.0],
        mst_master_omega=[1.0, 10.0, 100.0],
        mst_learner_weights=['all', 'infeasible'],
        lrn_warm_start=[False, True],
        mst_loss_fn=['mae', 'mse']
    )
    _study(regression, dataset='synthetic', name='regression', data_args=dict(full_grid=False))

    # STUDIES ON CLASSIFICATION DATASET
    fixed_parameters = dict(
        mst_backend='gurobi',
        mst_beta=None,
        mnt_kind='group',
        mt_init_step='pretraining',
        mst_learner_omega=1.0,
        mst_master_omega=1.0,
        mst_learner_weights='all',
        lrn_warm_start=False
    )
    classification = cartesian_product(
        fixed_parameters=fixed_parameters,
        mst_master_kind=['classification'],
        mst_loss_fn=['hd', 'bce', 'rbce', 'sbce'],
        lrn_loss=['binary_crossentropy'],
        mst_alpha=[0.01, 0.1, 1.0]
    ) + cartesian_product(
        fixed_parameters=fixed_parameters,
        mst_master_kind=['regression'],
        mst_loss_fn=['mae', 'mse'],
        lrn_loss=['binary_crossentropy'],
        mst_alpha=[0.01, 0.1, 1.0]
    ) + cartesian_product(
        fixed_parameters=fixed_parameters,
        mst_master_kind=['regression'],
        mst_loss_fn=['mae', 'mse'],
        lrn_loss=['mse'],
        mst_alpha=[0.01, 0.1, 1.0]
    )
    _study(classification, dataset='restaurants', name='classification', data_args=dict(full_grid=False))

    shutil.rmtree('wandb')
