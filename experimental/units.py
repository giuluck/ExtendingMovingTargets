"""Hidden Units Tuning Script."""

import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from experimental.utils import DatasetFactory

if __name__ == '__main__':
    datasets = DatasetFactory()
    study = [[]] + [[hu] * (lrs + 1) for hu in [8, 16, 32, 64, 128, 256] for lrs in range(6)]
    for ds in ['synthetic', 'puzzles', 'restaurants', 'default', 'law']:
        if ds in ['synthetic', 'restaurants']:
            factory, _ = datasets.get_dataset(name=ds, data_args=dict(grid_ground=2))
        else:
            factory, _ = datasets.get_dataset(name=ds, data_args=dict(full_features=False, grid_ground=2))
        print(f'DATASET: {ds.upper()}')
        for i, h_units in enumerate(study):
            print(f'Trial {i + 1:0{len(str(len(study)))}}/{len(study)}: {h_units}')
            factory.get_mlp(wandb_name=f'{ds}-slim: {h_units}', wandb_project='sc_units', h_units=h_units).validate()
        print('\n\n\n')
