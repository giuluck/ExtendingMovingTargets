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
        seed=[0, 1, 2],
        alpha=[0.01, 0.1, 1.0],
        learner_weights=['all', 'infeasible'],
        learner_omega=[1],
        master_omega=[1],
        warm_start=[True, False],
        dataset=['restaurants', 'default', 'law'],
        kind=[('classes', True), ('probabilities', 'bce')]
    )
    for s in study:
        kind, param = s['kind']
        if kind == 'probabilities':
            s['loss_fn'] = param
        elif kind == 'classes':
            s['use_prob'] = param
        else:
            raise ValueError(f"unknown kind '{kind}'")

    # begin study
    for i, config in enumerate(study):
        start_time = time.time()
        print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}', end=' ')
        dataset = config['dataset']
        del config['dataset']
        try:
            manager, _ = get_dataset(dataset=dataset)
            manager(
                seed=config['seed'],
                kind=config['kind'],
                warm_start=config['warm_start'],
                master_args={k: v for k, v in config.items() if k not in ['seed', 'kind', 'warm_start']}
            ).fit(
                iterations=20,
                verbose=False,
                callbacks=[WandBLogger(project='sc', entity='giuluck', run_name=dataset, crashed=False, **config)]
            )
            print(f'-- elapsed time: {time.time() - start_time}')
        except RuntimeError:
            print('-- unsolvable')
            WandBLogger.instance.config['crashed'] = True
            WandBLogger.instance.finish()
        except:
            print('-- errors')
shutil.rmtree('wandb')
