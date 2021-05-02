import numpy as np
import pandas as pd
from docplex.mp.model import Model as CPModel
from tensorflow.python.keras.callbacks import EarlyStopping

from moving_targets.masters import CplexMaster
from src import regressions as reg
# noinspection PyUnresolvedReferences
from moving_targets.callbacks import FileLogger
from moving_targets.metrics import R2, MSE, MAE, MonotonicViolation
from src.models import MT, MTMaster, MTLearner, MLP
from src.regressions.model import cars_summary, synthetic_summary, puzzles_summary, import_extension_methods
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
        dt = reg.load_cars('../../res/cars.csv', extrapolation=extra)
        xag, yag = dt['train']
        if ground is not None:
            xag, yag = xag.head(ground), yag.head(ground)
        mn = get_monotonicities_list(xag, lambda s, r: reg.compute_monotonicities(s, r, -1), 'sales', 'all', 'ignore')
        return xag, yag, mn, dt, np.zeros(len(yag)).astype(bool), cars_summary
    # datasets with augmentation
    dt, dirs, nrs, nas, fn = None, None, None, None, None
    if dataset == 'synthetic':
        dt = reg.load_synthetic(extrapolation=extra)
        dirs, nrs, nas, fn = [1, 0], 0, 15, synthetic_summary
    elif dataset == 'cars':
        dt = reg.load_cars('../../res/cars.csv', extrapolation=extra)
        dirs, nrs, nas, fn = -1, 0, 15, cars_summary
    elif dataset == 'puzzles':
        dt = reg.load_puzzles('../../res/puzzles.csv', extrapolation=extra)
        dirs, nrs, nas, fn = [-1, 1, 1], 465, [3, 4, 8], puzzles_summary
    xag, yag, fag = reg.get_augmented_data(
        x=dt['train'][0],
        y=dt['train'][1],
        directions=dirs,
        num_rand_samples=nrs if rand is None else rand,
        num_aug_samples=nas if aug is None else aug,
        num_ground_samples=ground
    )
    mn = get_monotonicities_list(
        data=fag,
        kinds=kinds,
        label=yag.columns[0],
        compute_monotonicities=lambda samples, references: reg.compute_monotonicities(samples, references, dirs)
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
        model.minimize(CplexMaster.mae_loss(model, yag, var))
        model.solve()
        # retrieve the optimal values and use it as labels
        yag = pd.Series(np.array([vy.solution_value for vy in var]), name=yag.name)
    return xag, yag, mn, dt, mask, fn


def neural_model():
    m = MLP(output_act=None, h_units=[16] * 4)
    m.compile(optimizer='adam', loss='mse')
    return m


if __name__ == '__main__':
    setup(seed=1)
    import_extension_methods()
    x_aug, y_aug, mono, data, aug_mk, summary = retrieve('cars', 'group', aug=None, ground=None, extra=False)

    # similar to the default behaviour of the scikit MLP (tol = 1e-4, n_iter_no_change = 10, max_iter = 200)
    es = EarlyStopping(monitor='loss', patience=10, min_delta=1e-4)
    learner = MTLearner(neural_model, epochs=200, callbacks=[es], verbose=False)
    master = MTMaster(monotonicities=mono, augmented_mask=aug_mk, loss_fn='mse', alpha=1.0,
                      learner_y='original', learner_weights='all', learner_omega=30, master_omega=1)
    iterations = 8

    num_col = int(np.ceil(np.sqrt(iterations + 1)))
    callbacks = [
        # FileLogger('temp/log.txt', routines=['on_iteration_end']),
        # ------------------------------------------------ SYNTHETIC ------------------------------------------------
        # DistanceAnalysis(data['scalers'], ground_only=True, num_columns=2, sorting_attribute='a'),
        # SyntheticAdjustments2D(data['scalers'], do_plot=False, file_signature='temp/synthetic_analysis'),
        # SyntheticAdjustments2D(data['scalers'], num_columns=num_col, sorting_attribute=None),
        # SyntheticAdjustments3D(data['scalers'], num_columns=num_col, sorting_attribute=None),
        # SyntheticResponse(data['scalers'], num_columns=num_col, sorting_attribute='a'),
        # ------------------------------------------------    CARS   ------------------------------------------------
        # DistanceAnalysis(data['scalers'], ground_only=True, num_columns=2, sorting_attribute='price'),
        CarsAdjustments(data['scalers'], do_plot=False, file_signature='temp/cars_analysis'),
        CarsAdjustments(data['scalers'], num_columns=num_col, sorting_attribute='price', plot_kind='scatter'),
        # ------------------------------------------------  PUZZLES  ------------------------------------------------
        # DistanceAnalysis(data['scalers'], ground_only=True, num_columns=2, sorting_attribute=None),
        # PuzzlesResponse(data['scalers'], feature='word_count', num_columns=num_col, sorting_attribute='word_count'),
        # PuzzlesResponse(data['scalers'], feature='star_rating', num_columns=num_col, sorting_attribute='star_rating'),
        # PuzzlesResponse(data['scalers'], feature='num_reviews', num_columns=num_col, sorting_attribute='num_reviews')
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
