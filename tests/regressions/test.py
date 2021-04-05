import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping

from src import regressions as reg
from moving_targets.callbacks import FileLogger
from moving_targets.metrics import R2
from src.models import MLP, MTLearner, MTMaster, MT
from src.util.augmentation import get_monotonicities_list

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
reg.import_extension_methods()
pd.options.display.max_rows = 10000
pd.options.display.max_columns = 10000
pd.options.display.width = 10000
pd.options.display.max_colwidth = 10000
pd.options.display.float_format = '{:.4f}'.format


def retrieve(dataset, kinds, num_ground_samples=None):
    # dataset without augmentation
    if dataset == 'cars univariate':
        MT.summary = MT.cars_summary
        data = reg.load_cars('../../res/cars.csv')
        xag, yag = data['train']
        mn = get_monotonicities_list(xag, lambda s, r: reg.compute_monotonicities(s, r, -1), 'sales', ['all'], 'ignore')
        return xag, yag, mn, {k: v for k, v in data.items() if k != 'scalers'}
    # datasets with augmentation
    data, dirs, nrs, nas = None, None, None, None
    if dataset == 'synthetic':
        MT.summary = MT.synthetic_summary
        data, dirs, nrs, nas = reg.load_synthetic(), [1, 0], 0, 15
    elif dataset == 'cars':
        MT.summary = MT.cars_summary
        data, dirs, nrs, nas = reg.load_cars('../../res/cars.csv'), -1, 0, 15
    elif dataset == 'puzzles':
        MT.summary = MT.puzzles_summary
        data, dirs, nrs, nas = reg.load_puzzles('../../res/puzzles.csv'), [-1, 1, 1], 465, [3, 4, 8]
    xag, yag, fag = reg.get_augmented_data(
        x=data['train'][0],
        y=data['train'][1],
        directions=dirs,
        num_random_samples=nrs,
        num_augmented_samples=nas,
        num_ground_samples=num_ground_samples
    )
    mono = get_monotonicities_list(
        data=fag,
        kinds=kinds,
        label=yag.columns[0],
        compute_monotonicities=lambda samples, references: reg.compute_monotonicities(samples, references, dirs)
    )
    return xag, yag[yag.columns[0]], mono, {k: (x, y) for k, (x, y) in data.items() if k != 'scalers'}


def get_model():
    model = MLP(output_act=None, h_units=[32, 32])
    model.compile(optimizer='adam', loss='mse')
    return model


class TestMTL(MTLearner):
    pass


class TestMTM(MTMaster):
    def adjust_targets(self, macs, x, y, iteration):
        return super(TestMTM, self).adjust_targets(macs, x, y, iteration)


class TestMT(MT):
    def log(self, **kwargs):
        kwargs = {k.replace('/', ': '): v for k, v in kwargs.items() if k != 'iteration' and 'time' not in k}
        self.cache.update(kwargs)


if __name__ == '__main__':
    x_aug, y_aug, monotonicities, val_data = retrieve('synthetic', 'group')

    # moving targets
    mt = TestMT(
        learner=TestMTL(
            build_model=get_model,
            restart_fit=True,
            callbacks=[EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)],
            epochs=0,
            verbose=0
        ),
        master=TestMTM(monotonicities, loss_fn='mae', alpha=1, beta=1),
        init_step='pretraining',
        metrics=[R2()]
    )
    mt.fit(
        x=x_aug.values,
        y=y_aug.values,
        iterations=1,
        val_data=val_data,
        callbacks=[FileLogger(), FileLogger('log.txt')]
    ).plot(figsize=(25, 15))
    mt.summary(**val_data)

    exit(0)
    plt.show()
