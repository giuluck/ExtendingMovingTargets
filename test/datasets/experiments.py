"""Experiments Script."""

import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import shutil

from moving_targets.callbacks import WandBLogger
from src.util.dictionaries import cartesian_product
from test.datasets.tests import get_dataset

if __name__ == '__main__':
    # create study list
    study = cartesian_product(
        mst_alpha=[0.01, 0.1, 1.0],
        dataset=['cars', 'synthetic', 'puzzles'],
        lrn_warm_start=[True, False],
        mst_learner_omega=[1, 10, 100],
        mst_master_omega=[1, 10, 100],
        mst_learner_weights=['all', 'infeasible'],
    )

    # begin study
    for i, config in enumerate(study):
        start_time = time.time()
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end=' ')
        dataset = config['dataset']
        del config['dataset']
        try:
            manager, _ = get_dataset(dataset=dataset)
            manager(seed=42, **config).validate(
                num_folds=5,
                iterations=20,
                verbose=False,
                callbacks=[WandBLogger(project='sc', entity='giuluck', run_name=dataset, crashed=False, **config)]
            )
            print(f'-- elapsed time: {time.time() - start_time}')
        except RuntimeError:
            print('-- unsolvable')
            WandBLogger.instance.config['crashed'] = True
            WandBLogger.instance.finish()
        # except:
        #     print('-- errors')
    shutil.rmtree('wandb')
