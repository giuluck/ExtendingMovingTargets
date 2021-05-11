import numpy as np

from moving_targets.callbacks import FileLogger
from test.datasets.managers import DistanceAnalysis, SyntheticAdjustments2D, SyntheticAdjustments3D, \
    SyntheticResponse, CarsAdjustments, PuzzlesResponse, CarsTest, CarsUnivariateTest, SyntheticTest, PuzzlesTest, \
    RestaurantsTest, RestaurantsAdjustment, LawTest, LawAdjustments, LawResponse, DefaultTest, DefaultAdjustments


def get_dataset(dataset, num_col=1):
    # DATASET AND CALLBACKS
    ds, cb = None, None
    if dataset == 'cars univariate':
        ds = CarsUnivariateTest
        cb = [
            FileLogger('temp/log.txt', routines=['on_iteration_end']),
            DistanceAnalysis(ground_only=True, num_columns=num_col, sorting_attribute='price'),
            CarsAdjustments(num_columns=num_col, sorting_attribute='price', plot_kind='scatter')
        ]
    elif dataset == 'cars':
        ds = CarsTest
        cb = [
            FileLogger('temp/log.txt', routines=['on_iteration_end']),
            DistanceAnalysis(ground_only=True, num_columns=num_col, sorting_attribute='price'),
            CarsAdjustments(num_columns=num_col, sorting_attribute='price', plot_kind='scatter')
        ]
    elif dataset == 'synthetic':
        ds = SyntheticTest
        cb = [
            FileLogger('temp/log.txt', routines=['on_iteration_end']),
            DistanceAnalysis(ground_only=True, num_columns=num_col, sorting_attribute='a'),
            SyntheticAdjustments2D(num_columns=num_col, sorting_attribute=None),
            SyntheticAdjustments3D(num_columns=num_col, sorting_attribute=None, data_points=True),
            SyntheticResponse(num_columns=num_col, sorting_attribute='a')
        ]
    elif dataset == 'puzzles':
        ds = PuzzlesTest
        cb = [
            FileLogger('temp/log.txt', routines=['on_iteration_end']),
            DistanceAnalysis(ground_only=True, num_columns=num_col, sorting_attribute=None),
            PuzzlesResponse(feature='word_count', num_columns=num_col, sorting_attribute='word_count'),
            PuzzlesResponse(feature='star_rating', num_columns=num_col, sorting_attribute='star_rating'),
            PuzzlesResponse(feature='num_reviews', num_columns=num_col, sorting_attribute='num_reviews')
        ]
    elif dataset == 'law':
        ds = LawTest
        cb = [
            FileLogger('temp/log.txt', routines=['on_iteration_end']),
            LawAdjustments(num_columns=num_col, sorting_attribute=None, data_points=True),
            LawResponse(feature='lsat', num_columns=num_col, sorting_attribute='lsat'),
            LawResponse(feature='ugpa', num_columns=num_col, sorting_attribute='ugpa')
        ]
    elif dataset == 'default':
        ds = DefaultTest
        cb = [
            FileLogger('temp/log.txt', routines=['on_iteration_end']),
            DefaultAdjustments(num_columns=num_col, sorting_attribute='payment')
        ]
    elif dataset == 'restaurants':
        ds = RestaurantsTest
        cb = [
            FileLogger('temp/log.txt', routines=['on_iteration_end']),
            RestaurantsAdjustment(rating='D', num_columns=num_col, data_points=False),
            RestaurantsAdjustment(rating='DD', num_columns=num_col, data_points=False),
            RestaurantsAdjustment(rating='DDD', num_columns=num_col, data_points=False),
            RestaurantsAdjustment(rating='DDDD', num_columns=num_col, data_points=False),
        ]
    # PLOT_ARGS
    pa = None
    if dataset in ['cars univariate', 'cars', 'synthetic', 'puzzles']:
        pa = dict(columns=[
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
    elif dataset in ['law', 'default']:
        pa = dict(columns=[
            'learner/loss',
            'metrics/train crossentropy',
            'metrics/train accuracy',
            'metrics/is feasible',
            'learner/epochs',
            'metrics/validation crossentropy',
            'metrics/validation accuracy',
            'metrics/pct. violation',
            'master/avg. flips',
            'metrics/test crossentropy',
            'metrics/test accuracy',
            'metrics/avg. violation'
        ])
    elif dataset in ['restaurants']:
        pa = dict(columns=[
            'learner/loss',
            'metrics/train crossentropy',
            'metrics/train auc',
            'metrics/is feasible',
            'learner/epochs',
            'metrics/validation crossentropy',
            'metrics/validation auc',
            'metrics/pct. violation',
            'master/avg. flips',
            'metrics/test crossentropy',
            'metrics/test auc',
            'metrics/avg. violation'
        ])
    return ds, cb, pa


if __name__ == '__main__':
    iterations = 1
    manager, callbacks, plot_args = get_dataset(dataset='law', num_col=int(np.ceil(np.sqrt(iterations + 1))))
    manager(
        warm_start=False,
        master_args=dict(
            alpha=1.0,
            use_prob=True,
            learner_weights='all',
            learner_omega=1.0,
            master_omega=1.0
        )
    ).fit(iterations=iterations, callbacks=callbacks, plot_args=plot_args, summary_args={})
