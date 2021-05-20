import numpy as np

from moving_targets.callbacks import FileLogger
from src.models import MTRegressionMaster
from test.datasets.managers import DistanceAnalysis, SyntheticAdjustments2D, SyntheticAdjustments3D, \
    SyntheticResponse, CarsAdjustments, PuzzlesResponse, CarsTest, CarsUnivariateTest, SyntheticTest, PuzzlesTest, \
    RestaurantsTest, RestaurantsAdjustment, LawTest, LawAdjustments, LawResponse, DefaultTest, DefaultAdjustments


def get_dataset(dataset, num_col=1, callback_functions=None):
    # DATASET AND CALLBACKS
    callback_functions = [] if callback_functions is None else callback_functions
    ds, cb = None, []
    if 'logger' in callback_functions:
        cb.append(FileLogger('temp/log.txt', routines=['on_iteration_end']))
    if dataset == 'cars univariate':
        ds = CarsUnivariateTest
        if 'distance' in callback_functions:
            cb.append(DistanceAnalysis(ground_only=True, num_columns=num_col, sorting_attribute='price',
                                       file_signature='temp/cars_univariate_distance'))
        if 'adjustments' in callback_functions:
            cb.append(CarsAdjustments(num_columns=num_col, sorting_attribute='price', plot_kind='scatter',
                                      file_signature='temp/cars_univariate_adjustments'))
    elif dataset == 'cars':
        ds = CarsTest
        if 'distance' in callback_functions:
            cb.append(DistanceAnalysis(ground_only=True, num_columns=num_col, sorting_attribute='price',
                                       file_signature='temp/cars_distance'))
        if 'adjustments' in callback_functions:
            cb.append(CarsAdjustments(num_columns=num_col, sorting_attribute='price', plot_kind='scatter',
                                      file_signature='temp/cars_adjustments'))
    elif dataset == 'synthetic':
        ds = SyntheticTest
        if 'distance' in callback_functions:
            cb.append(DistanceAnalysis(ground_only=True, num_columns=num_col, sorting_attribute='a',
                                       file_signature='temp/synthetic_distance'))
        if 'adjustments' in callback_functions:
            cb += [
                SyntheticAdjustments2D(num_columns=num_col, sorting_attribute=None,
                                       file_signature='temp/synthetic_adjustment_2D'),
                SyntheticAdjustments3D(num_columns=num_col, sorting_attribute=None, data_points=True,
                                       file_signature='temp/synthetic_adjustment_3D')
            ]
        if 'adjustments2D' in callback_functions:
            cb.append(SyntheticAdjustments2D(num_columns=num_col, sorting_attribute=None,
                                             file_signature='temp/synthetic_adjustment_2D'))
        if 'adjustments3D' in callback_functions:
            cb.append(SyntheticAdjustments3D(num_columns=num_col, sorting_attribute=None, data_points=True,
                                             file_signature='temp/synthetic_adjustment_3D'))
        if 'response' in callback_functions:
            cb.append(SyntheticResponse(num_columns=num_col, sorting_attribute='a',
                                        file_signature='temp/synthetic_response'))
    elif dataset == 'puzzles':
        ds = PuzzlesTest
        if 'distance' in callback_functions:
            cb.append(DistanceAnalysis(ground_only=True, num_columns=num_col, sorting_attribute=None,
                                       file_signature='temp/puzzles_distance'))
        if 'response' in callback_functions:
            cb += [
                PuzzlesResponse(feature='word_count', num_columns=num_col, sorting_attribute='word_count',
                                file_signature='temp/puzzles_response_word_count'),
                PuzzlesResponse(feature='star_rating', num_columns=num_col, sorting_attribute='star_rating',
                                file_signature='temp/puzzles_response_star_rating'),
                PuzzlesResponse(feature='num_reviews', num_columns=num_col, sorting_attribute='num_reviews',
                                file_signature='temp/puzzles_response_num_reviews')
            ]
    elif dataset == 'law':
        ds = LawTest
        if 'adjustments' in callback_functions:
            cb.append(LawAdjustments(num_columns=num_col, sorting_attribute=None, data_points=True,
                                     file_signature='temp/law_adjustments'))
        if 'response' in callback_functions:
            cb += [
                LawResponse(feature='lsat', num_columns=num_col, sorting_attribute='lsat',
                            file_signature='temp/law_response_lsat'),
                LawResponse(feature='ugpa', num_columns=num_col, sorting_attribute='ugpa',
                            file_signature='temp/law_response_ugpa')
            ]
    elif dataset == 'default':
        ds = DefaultTest
        if 'adjustments' in callback_functions:
            cb.append(DefaultAdjustments(num_columns=num_col, sorting_attribute='payment',
                                         file_signature='temp/default_adjustments'))
    elif dataset == 'restaurants':
        ds = RestaurantsTest
        if 'response' in callback_functions:
            cb += [
                RestaurantsAdjustment(rating='D', num_columns=num_col, data_points=False,
                                      file_signature='temp/restaurant_response_D'),
                RestaurantsAdjustment(rating='DD', num_columns=num_col, data_points=False,
                                      file_signature='temp/restaurant_response_DD'),
                RestaurantsAdjustment(rating='DDD', num_columns=num_col, data_points=False,
                                      file_signature='temp/restaurant_response_DDD'),
                RestaurantsAdjustment(rating='DDDD', num_columns=num_col, data_points=False,
                                      file_signature='temp/restaurant_response_DDDD'),
            ]
    return ds, cb


def get_plot_args(mng):
    return dict(columns=[
        'learner/loss',
        'metrics/train loss',
        'metrics/train metric',
        'metrics/is feasible',
        'learner/epochs',
        'metrics/validation loss',
        'metrics/validation metric',
        'metrics/pct. violation',
        f'master/{"adj. mse" if issubclass(mng.moving_targets.master.__class__, MTRegressionMaster) else "avg. flips"}',
        'metrics/test loss',
        'metrics/test metric',
        'metrics/avg. violation'
    ])


if __name__ == '__main__':
    iterations = 1
    manager, callbacks = get_dataset(
        dataset='restaurants',
        num_col=int(np.ceil(np.sqrt(iterations + 1))),
        callback_functions=['adjustments', 'response']
    )
    manager = manager(
        kind='classification',
        warm_start=False,
        master_args=dict(
            alpha=1.0,
            master_omega=1.0,
            learner_omega=1.0,
            learner_weights='all'
        )
    )
    manager.fit(iterations=iterations, callbacks=callbacks, plot_args=get_plot_args(manager), summary_args={})
