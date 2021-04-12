import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.neural_network import MLPRegressor
from tensorflow.python.keras.callbacks import EarlyStopping

from src import regressions as reg
from moving_targets.callbacks import FileLogger
from moving_targets.metrics import R2, MSE, MAE
from src.models import Model, MLP, MTLearner, MTMaster, MT
from src.util.augmentation import get_monotonicities_list
from cars_callbacks import CarsAdjustments, CarsBounds, CarsGround
from synthetic_callbacks import SyntheticAdjustments, SyntheticResponse, SyntheticGround, SyntheticBounds

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
        dt = reg.load_cars('../../res/cars.csv')
        xag, yag = dt['train']
        if ground is not None:
            xag, yag = xag.head(ground), yag.head(ground)
        mn = get_monotonicities_list(xag, lambda s, r: reg.compute_monotonicities(s, r, -1), 'sales', 'all', 'ignore')
        return xag, yag, mn, dt
    # datasets with augmentation
    dt, dirs, nrs, nas = None, None, None, None
    if dataset == 'synthetic':
        Model.evaluation_summary = Model.synthetic_summary
        dt, dirs, nrs, nas = reg.load_synthetic(), [1, 0], 0, 15
    elif dataset == 'cars':
        Model.evaluation_summary = Model.cars_summary
        dt, dirs, nrs, nas = reg.load_cars('../../res/cars.csv'), -1, 0, 15
    elif dataset == 'puzzles':
        Model.evaluation_summary = Model.puzzles_summary
        dt, dirs, nrs, nas = reg.load_puzzles('../../res/puzzles.csv'), [-1, 1, 1], 465, [3, 4, 8]
    xag, yag, fag = reg.get_augmented_data(
        x=dt['train'][0],
        y=dt['train'][1],
        directions=dirs,
        num_random_samples=nrs if rand is None else rand,
        num_augmented_samples=nas if aug is None else aug,
        num_ground_samples=ground
    )
    mn = get_monotonicities_list(
        data=fag,
        kinds=kinds,
        label=yag.columns[0],
        compute_monotonicities=lambda samples, references: reg.compute_monotonicities(samples, references, dirs)
    )
    return xag, yag[yag.columns[0]], mn, dt


class TestMTL(MTLearner):
    def __init__(self, backend='scikit', warm_start=False, verbose=False):
        if backend == 'scikit':
            def model():
                return MLPRegressor([16] * 4, solver='adam', warm_start=warm_start, verbose=verbose)

            super(TestMTL, self).__init__(model, warm_start=True)
        elif backend == 'keras':
            def model():
                m = MLP(output_act=None, h_units=[16] * 4)
                m.compile(optimizer='adam', loss='mse')
                return m

            # similar to the default behaviour of the scikit MLP:
            # > tol (min_delta) = 1e-4
            # > n_iter_no_change (patience) = 10
            # > max_iter (epochs) = 200
            fit_args = dict(
                callbacks=[EarlyStopping(monitor='loss', patience=10, min_delta=1e-4)],
                verbose=verbose,
                epochs=200
            )
            super(TestMTL, self).__init__(model, warm_start=warm_start, **fit_args)
        else:
            raise ValueError('Wrong backend')


class TestMTM(MTMaster):
    def __init__(self, monotonicities, loss_fn='mae', alpha=1., beta=1.):
        super(TestMTM, self).__init__(monotonicities=monotonicities, loss_fn=loss_fn, alpha=alpha, beta=beta)
        self.base_beta = beta

    def y_loss(self, macs, model, model_info, x, y, iteration):
        y_loss = super(TestMTM, self).y_loss(macs, model, model_info, x, y, iteration)
        self.beta = self.base_beta * y_loss
        return y_loss

    def is_feasible(self, macs, model, model_info, x, y, iteration):
        is_feasible = super(TestMTM, self).is_feasible(macs, model, model_info, x, y, iteration)
        return is_feasible

    def return_solutions(self, macs, solution, model_info, x, y, iteration):
        adj_y = super(TestMTM, self).return_solutions(macs, solution, model_info, x, y, iteration)
        # sample weights are directly proportional to the distance from the adjusted target to the prediction
        # (if adj_y == p then that sample is pretty useless)
        _, pred = model_info
        mask = np.isnan(y)
        sample_weight = np.abs(adj_y - pred)
        if sample_weight[mask].max() > 0:
            sample_weight = sample_weight / sample_weight[mask].max()
        sample_weight[~mask] = 1.0
        return adj_y, {'sample_weight': sample_weight}


class TestMT(MT):
    def on_iteration_start(self, macs, x, y, val_data, iteration):
        print(f'-------------------- ITERATION: {iteration:02} --------------------')
        super(TestMT, self).on_iteration_start(macs, x, y, val_data, iteration)

    def on_iteration_end(self, macs, x, y, val_data, iteration):
        super(TestMT, self).on_iteration_end(macs, x, y, val_data, iteration)
        print(f'Time: {self.cache["time/iteration"]:.4f} s')


if __name__ == '__main__':
    x_aug, y_aug, mono, data = retrieve('synthetic', 'group', aug=None, ground=None)
    # mono = []

    callbacks = [
        # CarsAdjustments(data['scalers'], num_columns=5, sorting_attributes='price', plot_kind='scatter'),
        # CarsBounds(data['scalers'], num_columns=1, sorting_attributes='price'),
        # CarsGround(data['scalers'], num_columns=1, sorting_attributes=None),
        # SyntheticAdjustments(data['scalers'], num_columns=5, sorting_attributes='a'),
        # SyntheticResponse(data['scalers'], num_columns=5, sorting_attributes='a'),
        SyntheticBounds(data['scalers'], num_columns=1, sorting_attributes='a'),
        SyntheticGround(data['scalers'], num_columns=1, sorting_attributes=None),
        FileLogger('../../temp/log.txt', routines=['on_iteration_end'])
    ]

    # moving targets
    mt = TestMT(
        learner=TestMTL(backend='keras', warm_start=False, verbose=False),
        master=TestMTM(monotonicities=mono, loss_fn='mae', alpha=1, beta=1),
        init_step='pretraining',
        metrics=[MSE(), MAE(), R2()]
    )
    history = mt.fit(
        x=x_aug,
        y=y_aug,
        iterations=1,
        val_data={k: v for k, v in data.items() if k != 'scalers'},
        callbacks=callbacks,
        verbose=0
    )

    history.plot(figsize=(20, 10), n_columns=4, columns=[
        'learner/loss',
        'learner/epochs',
        'metrics/train_r2',
        'master/is feasible',
        'metrics/train_mse',
        'metrics/train_mae',
        'metrics/validation_r2',
        'master/pct. violation',
        'master/adj. mse',
        'master/adj. mae',
        'metrics/test_r2',
        'master/avg. violation'
    ])

    exit()
    mt.evaluation_summary(**data)
