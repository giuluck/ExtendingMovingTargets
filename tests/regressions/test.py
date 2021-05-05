import numpy as np
import pandas as pd
from docplex.mp.model import Model as CPModel
from tensorflow.python.keras.callbacks import EarlyStopping

from datasets import Cars, Synthetic, Puzzles
from moving_targets.masters import CplexMaster
# noinspection PyUnresolvedReferences
from moving_targets.callbacks import FileLogger
from moving_targets.metrics import R2, MSE, MAE, MonotonicViolation
from src.models import MT, MTRegressionMaster, MTLearner
from src.util.augmentation import get_monotonicities_list
# noinspection PyUnresolvedReferences
from tests.regressions.callbacks import CarsAdjustments, SyntheticAdjustments2D, SyntheticAdjustments3D, \
    SyntheticResponse, PuzzlesResponse
# noinspection PyUnresolvedReferences
from tests.util.callbacks import BoundsAnalysis, DistanceAnalysis
from tests.util.experiments import setup


def retrieve(dataset, kinds, rand=None, aug=None, ground=None, extra=False, supervised=False):
    # dataset without augmentation
    if dataset == 'cars univariate':
        ds, sc = Cars()
        dt = ds.load_data(filepath='../../res/cars.csv', extrapolation=extra)
        xag, yag = dt['train']
        if ground is not None:
            xag, yag = xag.head(ground), yag.head(ground)
        mn = get_monotonicities_list(xag, lambda s, r: ds.compute_monotonicities(s, r), 'sales', 'all', 'ignore')
        return xag, yag, mn, dt, sc, np.zeros(len(yag)).astype(bool), ds.evaluation_summary
    # datasets with augmentation
    if dataset == 'synthetic':
        ds = Synthetic()
        dt, _ = ds.load_data(extrapolation=extra)
        nrs, nas, fn = 0, 15, ds.evaluation_summary
    elif dataset == 'cars':
        ds = Cars()
        dt, _ = ds.load_data(filepath='../../res/cars.csv', extrapolation=extra)
        nrs, nas, fn = 0, 15, ds.evaluation_summary
    elif dataset == 'puzzles':
        ds = Puzzles()
        dt, _ = ds.load_data(filepath='../../res/puzzles.csv', extrapolation=extra)
        nrs, nas, fn = 465, [3, 4, 8], ds.evaluation_summary
    else:
        raise ValueError(f"'{dataset}' is not a valid dataset")
    (xag, yag), sc = ds.get_augmented_data(
        x=dt['train'][0],
        y=dt['train'][1],
        num_augmented=nas if aug is None else aug,
        num_random=nrs if rand is None else rand,
        num_ground=ground
    )
    mn = get_monotonicities_list(
        data=pd.concat((xag, yag), axis=1),
        kind=kinds,
        label=yag.columns[0],
        compute_monotonicities=lambda samples, references: ds.compute_monotonicities(samples, references)
    )
    yag = yag[yag.columns[0]]
    mask = np.isnan(yag)
    if supervised:
        # assign random values to unlabeled data
        yag[mask] = np.random.uniform(low=yag[~mask].min(), high=yag[~mask].max(), size=len(yag[mask]))
        # basically a pretraining step in which we force the original data not to be changed
        model = CPModel()
        var = np.array(model.continuous_var_list(keys=len(yag), name='y'))
        model.add_constraints([var[hi] >= var[li] for hi, li in mn])
        model.add_constraints([v == y for v, y in zip(var[~mask], yag[~mask])])
        model.minimize(CplexMaster.mean_squared_error(model, yag, var))
        model.solve()
        # retrieve the optimal values and use it as labels
        yag = pd.Series(np.array([vy.solution_value for vy in var]), name=yag.name)
    return xag, yag, mn, dt, sc, mask, fn


if __name__ == '__main__':
    setup(seed=1)
    x_aug, y_aug, mono, data, scalers, aug_mk, summary = retrieve('cars', 'group', aug=None, extra=False)

    # similar to the default behaviour of the scikit MLP (tol = 1e-4, n_iter_no_change = 10, max_iter = 200)
    es = EarlyStopping(monitor='loss', patience=10, min_delta=1e-4)
    learner = MTLearner(output_act=None, h_units=[16] * 4, optimizer='adam', loss='mse', scalers=scalers,
                        warm_start=False, epochs=200, callbacks=[es], verbose=False)
    master = MTRegressionMaster(monotonicities=mono, augmented_mask=aug_mk, loss_fn='mean_squared_error', alpha=0.01,
                                learner_y='original', learner_weights='all', learner_omega=1, master_omega=1)
    iterations = 8

    num_col = int(np.ceil(np.sqrt(iterations + 1)))
    callbacks = [
        # FileLogger('temp/log.txt', routines=['on_iteration_end']),
        # ------------------------------------------------ SYNTHETIC ------------------------------------------------
        # DistanceAnalysis(ground_only=True, num_columns=num_col, sorting_attribute='a'),
        # SyntheticAdjustments2D(do_plot=False, file_signature='temp/synthetic_analysis'),
        # SyntheticAdjustments2D(num_columns=num_col, sorting_attribute=None),
        # SyntheticAdjustments3D(num_columns=num_col, sorting_attribute=None),
        # SyntheticResponse(num_columns=num_col, sorting_attribute='a'),
        # ------------------------------------------------    CARS   ------------------------------------------------
        # DistanceAnalysis(ground_only=True, num_columns=num_col, sorting_attribute='price'),
        # CarsAdjustments(do_plot=False, file_signature='temp/cars_analysis'),
        CarsAdjustments(num_columns=num_col, sorting_attribute='price', plot_kind='scatter'),
        # ------------------------------------------------  PUZZLES  ------------------------------------------------
        # DistanceAnalysis(ground_only=True, num_columns=num_col, sorting_attribute=None),
        # PuzzlesResponse(feature='word_count', num_columns=num_col, sorting_attribute='word_count'),
        # PuzzlesResponse(feature='star_rating', num_columns=num_col, sorting_attribute='star_rating'),
        # PuzzlesResponse(feature='num_reviews', num_columns=num_col, sorting_attribute='num_reviews')
    ]

    # moving targets
    mt = MT(
        learner=learner,
        master=master,
        init_step='pretraining',
        metrics=[MSE(), MAE(), R2(),
                 MonotonicViolation(monotonicities=mono, aggregation='average', name='avg. violation'),
                 MonotonicViolation(monotonicities=mono, aggregation='percentage', name='pct. violation'),
                 MonotonicViolation(monotonicities=mono, aggregation='feasible', name='is feasible')]
    )
    history = mt.fit(
        x=x_aug,
        y=y_aug,
        iterations=iterations,
        val_data={k: v for k, v in data.items() if k != 'scalers'},
        callbacks=callbacks,
        verbose=1
    )

    # exit()
    history.plot(figsize=(20, 10), n_columns=4, columns=[
        'learner/loss',
        'learner/epochs',
        'metrics/train r2',
        'metrics/is feasible',
        'metrics/train mse',
        'metrics/train mae',
        'metrics/validation r2',
        'metrics/pct. violation',
        'master/adj. mse',
        'master/adj. mae',
        'metrics/test r2',
        'metrics/avg. violation'
    ])

    # exit()
    print('-------------------------------------------------------')
    summary(mt, **data)
