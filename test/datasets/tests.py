import numpy as np

from moving_targets.callbacks import FileLogger
from test.datasets.managers import DistanceAnalysis, SyntheticAdjustments2D, SyntheticAdjustments3D, \
    SyntheticResponse, CarsAdjustments, PuzzlesResponse, CarsTest, CarsUnivariateTest, SyntheticTest, PuzzlesTest, \
    RestaurantsTest, RestaurantsResponse


def get_dataset(dataset, num_col=1, **kwargs):
    ds, cb, pa = None, None, None
    if dataset in ['cars univariate', 'cars', 'synthetic', 'puzzles']:
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
                SyntheticAdjustments3D(num_columns=num_col, sorting_attribute=None),
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
    elif dataset == 'restaurants':
        ds = RestaurantsTest
        cb = [
            FileLogger('temp/log.txt', routines=['on_iteration_end']),
            DistanceAnalysis(ground_only=True, num_columns=num_col, sorting_attribute=None),
            RestaurantsResponse(rating='D', num_columns=num_col),
            RestaurantsResponse(rating='DD', num_columns=num_col),
            RestaurantsResponse(rating='DDD', num_columns=num_col),
            RestaurantsResponse(rating='DDDD', num_columns=num_col),
        ]
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
    return ds(**kwargs), cb, pa


if __name__ == '__main__':
    iterations = 3
    manager, callbacks, plot_args = get_dataset(dataset='puzzles', num_col=int(np.ceil(np.sqrt(iterations + 1))))
    manager.fit(iterations=iterations, callbacks=callbacks, plot_args=plot_args, summary_args={})
