"""Datasets Factory Handlers."""

from typing import Optional, List, Tuple

from experimental.utils import DistanceAnalysis, CarsAdjustments, SyntheticAdjustments2D, SyntheticAdjustments3D, \
    SyntheticResponse, PuzzlesResponse, RestaurantsAdjustment, DefaultAdjustments, LawAdjustments, LawResponse
from experimental.utils.handlers import HandlersFactory, RegressionFactory, ClassificationFactory
from moving_targets.callbacks import Callback, FileLogger
from moving_targets.metrics import Accuracy, AUC
from src.datasets import CarsManager, DefaultManager, PuzzlesManager, SyntheticManager, RestaurantsManager, LawManager
from src.util.typing import Augmented


# noinspection PyMissingOrEmptyDocstring
class DatasetFactory:
    def __init__(self, res_folder: Optional[str] = '../res/', temp_folder: Optional[str] = '../temp/'):
        self.res_folder: Optional[str] = res_folder.strip('/')
        self.temp_folder: Optional[str] = temp_folder.strip('/')

    def _get_shared_callbacks(self, callbacks: Optional[List[str]]):
        callbacks: List[str] = [] if callbacks is None else callbacks
        cb: List[Callback] = []
        if 'logger' in callbacks:
            cb.append(FileLogger(f'{self.temp_folder}/log.txt', routines=['on_pretraining_end', 'on_iteration_end']))
        return cb, callbacks

    def get_dataset(self, name: str, **kwargs) -> Tuple[HandlersFactory, List[Callback]]:
        method = getattr(self, name.replace(' ', '_'))
        return method(**kwargs)

    def cars_univariate(self, data_args: Optional[dict] = None, h_units: tuple = (128, 128), num_col: int = 1,
                        callbacks: Optional[List[str]] = None, **kwargs) -> Tuple[RegressionFactory, List[Callback]]:
        cb, callbacks = self._get_shared_callbacks(callbacks)
        if 'distance' in callbacks:
            cb.append(DistanceAnalysis(ground_only=True, num_columns=num_col, sorting_attribute='price',
                                       file_signature=f'{self.temp_folder}/cars_univariate_distance'))
        if 'adjustments' in callbacks:
            cb.append(CarsAdjustments(num_columns=num_col, sorting_attribute='price', plot_kind='line',
                                      file_signature=f'{self.temp_folder}/cars_univariate_adjustments'))
        data_args = {} if data_args is None else data_args
        ds = RegressionFactory(manager=CarsManager(filepath=f'{self.res_folder}/cars.csv', **data_args),
                               h_units=h_units, num_augmented=0, monotonicities='all', errors='ignore', **kwargs)
        return ds, cb

    def cars(self, data_args: Optional[dict] = None, h_units: tuple = (128, 128), num_col: int = 1,
             callbacks: Optional[List[str]] = None, **kwargs) -> Tuple[RegressionFactory, List[Callback]]:
        cb, callbacks = self._get_shared_callbacks(callbacks)
        if 'distance' in callbacks:
            cb.append(DistanceAnalysis(ground_only=True, num_columns=num_col, sorting_attribute='price',
                                       file_signature=f'{self.temp_folder}/cars_distance'))
        if 'adjustments' in callbacks:
            cb.append(CarsAdjustments(num_columns=num_col, sorting_attribute='price', plot_kind='scatter',
                                      file_signature=f'{self.temp_folder}/cars_adjustments'))
        data_args = {} if data_args is None else data_args
        ds = RegressionFactory(manager=CarsManager(filepath=f'{self.res_folder}/cars.csv', **data_args),
                               h_units=h_units, **kwargs)
        return ds, cb

    def synthetic(self, data_args: Optional[dict] = None, h_units: tuple = (128, 128), num_col: int = 1,
                  callbacks: Optional[List[str]] = None, **kwargs) -> Tuple[RegressionFactory, List[Callback]]:
        cb, callbacks = self._get_shared_callbacks(callbacks)
        if 'distance' in callbacks:
            cb.append(DistanceAnalysis(ground_only=True, num_columns=num_col, sorting_attribute='a',
                                       file_signature=f'{self.temp_folder}/synthetic_distance'))
        if 'adjustments' in callbacks or 'adjustments2D' in callbacks:
            cb.append(SyntheticAdjustments2D(num_columns=num_col, sorting_attribute=None,
                                             file_signature=f'{self.temp_folder}/synthetic_adjustment_2D'))
        if 'adjustments' in callbacks or 'adjustments3D' in callbacks:
            cb.append(SyntheticAdjustments3D(num_columns=num_col, sorting_attribute=None, data_points=True,
                                             file_signature=f'{self.temp_folder}/synthetic_adjustment_3D'))
        if 'response' in callbacks:
            cb.append(SyntheticResponse(num_columns=num_col, sorting_attribute='a',
                                        file_signature=f'{self.temp_folder}/synthetic_response'))
        data_args = {} if data_args is None else data_args
        ds = RegressionFactory(manager=SyntheticManager(**data_args), h_units=h_units, **kwargs)
        return ds, cb

    def puzzles(self, data_args: Optional[dict] = None, h_units: tuple = (128, 128),
                num_col: int = 1, num_augmented: Augmented = (3, 4, 8), num_random: int = 465,
                callbacks: Optional[List[str]] = None, **kwargs) -> Tuple[RegressionFactory, List[Callback]]:
        cb, callbacks = self._get_shared_callbacks(callbacks)
        if 'distance' in callbacks:
            cb.append(DistanceAnalysis(ground_only=True, num_columns=num_col, sorting_attribute=None,
                                       file_signature=f'{self.temp_folder}/puzzles_distance'))
        if 'response' in callbacks:
            cb += [
                PuzzlesResponse(feature='word_count', num_columns=num_col, sorting_attribute='word_count',
                                file_signature=f'{self.temp_folder}/puzzles_response_word_count'),
                PuzzlesResponse(feature='star_rating', num_columns=num_col, sorting_attribute='star_rating',
                                file_signature=f'{self.temp_folder}/puzzles_response_star_rating'),
                PuzzlesResponse(feature='num_reviews', num_columns=num_col, sorting_attribute='num_reviews',
                                file_signature=f'{self.temp_folder}/puzzles_response_num_reviews')
            ]
        data_args = {} if data_args is None else data_args
        ds = RegressionFactory(manager=PuzzlesManager(filepath=f'{self.res_folder}/puzzles.csv', **data_args),
                               h_units=h_units, num_augmented=num_augmented, num_random=num_random, **kwargs)
        return ds, cb

    def restaurants(self, data_args: Optional[dict] = None, h_units: tuple = (128, 128), num_col: int = 1,
                    callbacks: Optional[List[str]] = None, **kwargs) -> Tuple[ClassificationFactory, List[Callback]]:
        cb, callbacks = self._get_shared_callbacks(callbacks)
        if 'response' in callbacks:
            cb += [
                RestaurantsAdjustment(rating='D', num_columns=num_col, data_points=False,
                                      file_signature=f'{self.temp_folder}/restaurant_response_D'),
                RestaurantsAdjustment(rating='DD', num_columns=num_col, data_points=False,
                                      file_signature=f'{self.temp_folder}/restaurant_response_DD'),
                RestaurantsAdjustment(rating='DDD', num_columns=num_col, data_points=False,
                                      file_signature=f'{self.temp_folder}/restaurant_response_DDD'),
                RestaurantsAdjustment(rating='DDDD', num_columns=num_col, data_points=False,
                                      file_signature=f'{self.temp_folder}/restaurant_response_DDDD'),
            ]
        ds = ClassificationFactory(
            manager=RestaurantsManager(**{} if data_args is None else data_args),
            mt_evaluation_metric=AUC(name='metric'),
            h_units=h_units,
            **kwargs
        )
        return ds, cb

    def default(self, data_args: Optional[dict] = None, h_units: tuple = (128, 128), num_col: int = 1,
                callbacks: Optional[List[str]] = None, **kwargs) -> Tuple[ClassificationFactory, List[Callback]]:
        cb, callbacks = self._get_shared_callbacks(callbacks)
        if 'adjustments' in callbacks:
            cb.append(DefaultAdjustments(num_columns=num_col, sorting_attribute='payment',
                                         file_signature=f'{self.temp_folder}/default_adjustments'))
        ds = ClassificationFactory(
            manager=DefaultManager(filepath=f'{self.res_folder}/default.csv', **{} if data_args is None else data_args),
            mt_evaluation_metric=Accuracy(name='metric'),
            h_units=h_units,
            **kwargs
        )
        return ds, cb

    def law(self, data_args: Optional[dict] = None, h_units: tuple = (128, 128), num_col: int = 1,
            callbacks: Optional[List[str]] = None, **kwargs) -> Tuple[ClassificationFactory, List[Callback]]:
        cb, callbacks = self._get_shared_callbacks(callbacks)
        if 'adjustments' in callbacks:
            cb.append(LawAdjustments(num_columns=num_col, sorting_attribute=None, data_points=True,
                                     file_signature=f'{self.temp_folder}/law_adjustments'))
        if 'response' in callbacks:
            cb += [
                LawResponse(feature='lsat', num_columns=num_col, sorting_attribute='lsat',
                            file_signature=f'{self.temp_folder}/law_response_lsat'),
                LawResponse(feature='ugpa', num_columns=num_col, sorting_attribute='ugpa',
                            file_signature=f'{self.temp_folder}/law_response_ugpa')
            ]
        ds = ClassificationFactory(
            manager=LawManager(filepath=f'{self.res_folder}/law.csv', **{} if data_args is None else data_args),
            mt_evaluation_metric=Accuracy(name='metric'),
            h_units=h_units,
            **kwargs
        )
        return ds, cb
