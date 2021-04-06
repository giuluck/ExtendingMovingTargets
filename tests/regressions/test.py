import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping

from src import regressions as reg
from moving_targets.callbacks import FileLogger
from moving_targets.metrics import R2, MSE
from src.models import Model, MLP, MTLearner, MTMaster, MT
from src.util.augmentation import get_monotonicities_list
from analysis_callback import CarsCallback

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
reg.import_extension_methods()
pd.options.display.max_rows = 10000
pd.options.display.max_columns = 10000
pd.options.display.width = 10000
pd.options.display.max_colwidth = 10000
pd.options.display.float_format = '{:.4f}'.format


def retrieve(dataset, kinds, rand=None, aug=None, ground=None):
    # dataset without augmentation
    if dataset == 'cars univariate':
        Model.evaluation_summary = Model.cars_summary
        data = reg.load_cars('../../res/cars.csv')
        xag, yag = data['train']
        if ground is not None:
            xag, yag = xag.head(ground), yag.head(ground)
        mn = get_monotonicities_list(xag, lambda s, r: reg.compute_monotonicities(s, r, -1), 'sales', 'all', 'ignore')
        return xag, yag, mn, data
    # datasets with augmentation
    data, dirs, nrs, nas = None, None, None, None
    if dataset == 'synthetic':
        Model.evaluation_summary = Model.synthetic_summary
        data, dirs, nrs, nas = reg.load_synthetic(), [1, 0], 0, 15
    elif dataset == 'cars':
        Model.evaluation_summary = Model.cars_summary
        data, dirs, nrs, nas = reg.load_cars('../../res/cars.csv'), -1, 0, 15
    elif dataset == 'puzzles':
        Model.evaluation_summary = Model.puzzles_summary
        data, dirs, nrs, nas = reg.load_puzzles('../../res/puzzles.csv'), [-1, 1, 1], 465, [3, 4, 8]
    xag, yag, fag = reg.get_augmented_data(
        x=data['train'][0],
        y=data['train'][1],
        directions=dirs,
        num_random_samples=nrs if rand is None else rand,
        num_augmented_samples=nas if aug is None else aug,
        num_ground_samples=ground
    )
    mono = get_monotonicities_list(
        data=fag,
        kinds=kinds,
        label=yag.columns[0],
        compute_monotonicities=lambda samples, references: reg.compute_monotonicities(samples, references, dirs)
    )
    return xag, yag[yag.columns[0]], mono, data


def get_model():
    model = MLP(output_act=None, h_units=[32, 32])
    model.compile(optimizer='adam', loss='mse')
    return model


class TestMTL(MTLearner):
    pass


class TestMTM(MTMaster):
    pass


class TestMT(MT):
    def log(self, **kwargs):
        kwargs = {k.replace('/', ': '): v for k, v in kwargs.items() if k not in ['time/learner', 'time/master']}
        self.cache.update(kwargs)


if __name__ == '__main__':
    x_aug, y_aug, monotonicities, val_data = retrieve('cars univariate', 'group', ground=None)

    # moving targets
    mt = TestMT(
        learner=TestMTL(
            build_model=get_model,
            restart_fit=True,
            callbacks=[EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)],
            epochs=1000,
            verbose=1
        ),
        master=TestMTM(monotonicities, loss_fn='mae', alpha=1, beta=1),
        init_step='pretraining',
        metrics=[MSE(), R2()]
    )
    history = mt.fit(
        x=x_aug.values,
        y=y_aug.values,
        iterations=9,
        val_data={k: v for k, v in val_data.items() if k != 'scalers'},
        callbacks=[CarsCallback(3, figsize=(24, 12)), FileLogger('log.txt')],
        verbose=1
    )
    plt.show()

    exit(0)
    history.plot(figsize=(25, 15))
    mt.evaluation_summary(**val_data)
    plt.show()
