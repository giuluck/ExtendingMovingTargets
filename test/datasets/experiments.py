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
    master_args = cartesian_product(
        alpha=[0.01, 0.1, 1.0],
        use_prob=[True, False],
        learner_weights=['all', 'infeasible'],
        master_omega=[1],
        learner_omega=[1]
    )
    study = cartesian_product(
        seed=[0, 1, 2],
        master_args=master_args,
        warm_start=[True, False],
        dataset=['lo', 'default', 'restaurants']
    )

    # begin study
    for i, p in enumerate(study):
        start_time = time.time()
        manager, _, _ = get_dataset(**p)
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end='')
        # noinspection PyBroadException
        try:
            manager.fit(
                iterations=20,
                verbose=False,
                callbacks=[WandBLogger(project='sc', entity='giuluck', run_name=p['dataset'], **p)]
            )
            print(f' -- elapsed time: {time.time() - start_time}')
        except RuntimeError:
            print(' -- unsolvable')
            WandBLogger.instance.config.update({'crashed': True})
            WandBLogger.instance.finish()
        except:
            print(' -- errors')
            WandBLogger.instance.config.update({'crashed': True})
            WandBLogger.instance.finish()
    shutil.rmtree('wandb')
