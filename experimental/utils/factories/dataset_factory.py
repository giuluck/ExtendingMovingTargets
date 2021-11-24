"""Datasets Factory Handlers."""

from typing import Optional, List, Tuple

from experimental.utils import DistanceAnalysis, CarsAdjustments, SyntheticAdjustments2D, SyntheticAdjustments3D, \
    SyntheticResponse, PuzzlesResponse, RestaurantsAdjustment, DefaultAdjustments, LawAdjustments, LawResponse
from experimental.utils.factories.handlers_factory import HandlersFactory
from moving_targets.callbacks import Callback, FileLogger
from moving_targets.metrics import Accuracy, AUC, MSE, R2, CrossEntropy
from src.datasets import CarsManager, DefaultManager, PuzzlesManager, SyntheticManager, RestaurantsManager, LawManager
from src.util.typing import Augmented


class DatasetFactory:
    """Factory class that returns dataset handlers."""

    def __init__(self, res_folder: Optional[str] = '../../res/', temp_folder: Optional[str] = '../../temp/'):
        """
        :param res_folder:
            The res folder filepath where to retrieve inputs.

        :param temp_folder:
            The temp folder filepath where to place outputs.
        """

        self.res_folder: Optional[str] = res_folder.strip('/')
        """The res folder filepath where to retrieve inputs."""

        self.temp_folder: Optional[str] = temp_folder.strip('/')
        """The temp folder filepath where to place outputs."""

    def _get_shared_callbacks(self, callbacks: Optional[List[str]]) -> Tuple[List[Callback],  List[str]]:
        """Checks if the input list contains the name of some shared callbacks and includes them in the output list.

        :param callbacks:
            List of callback names.

        :return:
            A tuple where the first item is a list of `Callback` object which are shared among datasets, and the second
            item is the same input list or an empty list if the input was None.
        """
        callbacks: List[str] = [] if callbacks is None else callbacks
        cb: List[Callback] = []
        if 'logger' in callbacks:
            cb.append(FileLogger(f'{self.temp_folder}/log.txt', routines=['on_pretraining_end', 'on_iteration_end']))
        return cb, callbacks

    def get_dataset(self, name: str, **dataset_kwargs) -> Tuple[HandlersFactory, List[Callback]]:
        """Gets a dataset handler by dataset name.

        :param name:
            The dataset name.

        :param dataset_kwargs:
            The dataset-dependent kwargs.

        :return:
            A tuple where the first item is the dataset handler and the second item is the list of its callbacks.
        """
        method = getattr(self, name.replace(' ', '_'))
        return method(**dataset_kwargs)

    def cars_univariate(self,
                        data_args: Optional[dict] = None,
                        h_units: Tuple[int] = (128, 128),
                        num_col: int = 1,
                        callbacks: Optional[List[str]] = None,
                        **kwargs) -> Tuple[HandlersFactory, List[Callback]]:
        """Builds an handlers factory for the cars univariate dataset.

        :param data_args:
            Custom arguments passed to a `CarsManager` instance.

            - full_grid: bool = False
            - extrapolation: bool = False
            - grid_augmented: int = 150
            - grid_ground: Optional[int] = None
            - x_scaling: Method = 'std'
            - y_scaling: Method = 'norm'
            - bound: Tuple[float, float] = (0, 100)

        :param h_units:
            The neural network/learner list of hidden units (ignored for TFL).

        :param num_col:
            Number of columns in the final subplot.

        :param callbacks:
            List of callbacks aliases.

        :param kwargs:
            Custom arguments passed to a `HandlersFactory` instance.

            - dataset: Optional[str] = None
            - wandb_project: Optional[str] = 'moving_targets'
            - wandb_entity: Optional[str] = 'giuluck'
            - seed: int = 0
            - optimizer: str = 'adam'
            - h_units: Optional[List[int]] = None
            - epochs: int = 1000
            - batch_size: int = 32
            - validation_split: float = 0.2
            - callbacks: Optional[List[Callback]] = None
            - verbose: bool = False
            - num_random: int = 0
            - num_ground: Optional[int] = None

        :return:
            A tuple containing the `HandlersFactory` as first item, and a list of `Callback` objects as second one.
        """
        cb, callbacks = self._get_shared_callbacks(callbacks)
        if 'distance' in callbacks:
            cb.append(DistanceAnalysis(ground_only=True, num_columns=num_col, sorting_attribute='price',
                                       file_signature=f'{self.temp_folder}/cars_univariate_distance'))
        if 'adjustments' in callbacks:
            cb.append(CarsAdjustments(num_columns=num_col, sorting_attribute='price', plot_kind='line',
                                      file_signature=f'{self.temp_folder}/cars_univariate_adjustments'))
        data_args = {} if data_args is None else data_args
        ds = HandlersFactory(manager=CarsManager(filepath=f'{self.res_folder}/cars.csv',
                                                 full_features=False,
                                                 **data_args),
                             master_kind='regression',
                             mt_metrics=[MSE(name='loss'), R2(name='metric')],
                             loss='mse',
                             output_act=None,
                             h_units=list(h_units),
                             num_augmented=0,
                             monotonicities='all',
                             errors='ignore',
                             **kwargs)
        return ds, cb

    def cars(self,
             data_args: Optional[dict] = None,
             h_units: tuple = (128, 128),
             num_col: int = 1,
             callbacks: Optional[List[str]] = None,
             **kwargs) -> Tuple[HandlersFactory, List[Callback]]:
        """Builds an handlers factory for the cars dataset.

        :param data_args:
            Custom arguments passed to a `CarsManager` instance.

            - full_features: bool = False
            - full_grid: bool = False
            - extrapolation: bool = False
            - grid_augmented: int = 150
            - grid_ground: Optional[int] = None
            - x_scaling: Method = 'std'
            - y_scaling: Method = 'norm'
            - bound: Tuple[float, float] = (0, 100)

        :param h_units:
            The neural network/learner list of hidden units (ignored for TFL).

        :param num_col:
            Number of columns in the final subplot.

        :param callbacks:
            List of callbacks aliases.

        :param kwargs:
            Custom arguments passed to a `HandlersFactory` instance.

            - dataset: Optional[str] = None
            - wandb_project: Optional[str] = 'moving_targets'
            - wandb_entity: Optional[str] = 'giuluck'
            - seed: int = 0
            - optimizer: str = 'adam'
            - h_units: Optional[List[int]] = None
            - epochs: int = 1000
            - batch_size: int = 32
            - validation_split: float = 0.2
            - callbacks: Optional[List[Callback]] = None
            - verbose: bool = False
            - num_augmented: Optional[int] = None
            - num_random: int = 0
            - num_ground: Optional[int] = None
            - monotonicities: str = 'group'
            - errors: str = 'raise'

        :return:
            A tuple containing the `HandlersFactory` as first item, and a list of `Callback` objects as second one.
        """
        cb, callbacks = self._get_shared_callbacks(callbacks)
        if 'distance' in callbacks:
            cb.append(DistanceAnalysis(ground_only=True, num_columns=num_col, sorting_attribute='price',
                                       file_signature=f'{self.temp_folder}/cars_distance'))
        if 'adjustments' in callbacks:
            cb.append(CarsAdjustments(num_columns=num_col, sorting_attribute='price', plot_kind='scatter',
                                      file_signature=f'{self.temp_folder}/cars_adjustments'))
        data_args = {} if data_args is None else data_args
        ds = HandlersFactory(manager=CarsManager(filepath=f'{self.res_folder}/cars.csv', **data_args),
                             master_kind='regression',
                             mt_metrics=[MSE(name='loss'), R2(name='metric')],
                             loss='mse',
                             output_act=None,
                             h_units=list(h_units),
                             **kwargs)
        return ds, cb

    def synthetic(self,
                  data_args: Optional[dict] = None,
                  h_units: tuple = (128, 128),
                  num_col: int = 1,
                  callbacks: Optional[List[str]] = None,
                  **kwargs) -> Tuple[HandlersFactory, List[Callback]]:
        """Builds an handlers factory for the synthetic dataset.

        :param data_args:
            Custom arguments passed to a `SyntheticManager` instance.

            - full_features: bool = False
            - full_grid: bool = False
            - extrapolation: bool = False
            - grid_augmented: int = 35
            - grid_ground: Optional[int] = None
            - x_scaling: Methods = 'std'
            - y_scaling: Methods = 'norm'
            - bound: Optional[Bound] = None

        :param h_units:
            The neural network/learner list of hidden units (ignored for TFL).

        :param num_col:
            Number of columns in the final subplot.

        :param callbacks:
            List of callbacks aliases.

        :param kwargs:
            Custom arguments passed to a `HandlersFactory` instance.

            - dataset: Optional[str] = None
            - wandb_project: Optional[str] = 'moving_targets'
            - wandb_entity: Optional[str] = 'giuluck'
            - seed: int = 0
            - optimizer: str = 'adam'
            - h_units: Optional[List[int]] = None
            - epochs: int = 1000
            - batch_size: int = 32
            - validation_split: float = 0.2
            - callbacks: Optional[List[Callback]] = None
            - verbose: bool = False
            - num_augmented: Optional[int] = None
            - num_random: int = 0
            - num_ground: Optional[int] = None
            - monotonicities: str = 'group'
            - errors: str = 'raise'

        :return:
            A tuple containing the `HandlersFactory` as first item, and a list of `Callback` objects as second one.
        """
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
        ds = HandlersFactory(manager=SyntheticManager(**data_args),
                             master_kind='regression',
                             mt_metrics=[MSE(name='loss'), R2(name='metric')],
                             loss='mse',
                             output_act=None,
                             h_units=list(h_units),
                             **kwargs)
        return ds, cb

    def puzzles(self,
                data_args: Optional[dict] = None,
                h_units: tuple = (128, 128),
                num_col: int = 1,
                num_augmented: Augmented = (3, 4, 8),
                num_random: int = 465,
                callbacks: Optional[List[str]] = None,
                **kwargs) -> Tuple[HandlersFactory, List[Callback]]:
        """Builds an handlers factory for the puzzles dataset.

        :param data_args:
            Custom arguments passed to a `PuzzlesManager` instance.

            - full_features: bool = False
            - full_grid: bool = False
            - extrapolation: bool = False
            - grid_augmented: int = 35
            - grid_ground: Optional[int] = None
            - x_scaling: Methods = 'std'
            - y_scaling: Methods = 'norm'
            - bound: Optional[Bound] = None

        :param h_units:
            The neural network/learner list of hidden units (ignored for TFL).

        :param num_col:
            Number of columns in the final subplot.

        :param callbacks:
            List of callbacks aliases.

        :param num_augmented:
            The number of augmented samples.

        :param num_random:
            The number of unlabelled random samples added to the original dataset.

        :param kwargs:
            Custom arguments passed to a `HandlersFactory` instance.

            - dataset: Optional[str] = None
            - wandb_project: Optional[str] = 'moving_targets'
            - wandb_entity: Optional[str] = 'giuluck'
            - seed: int = 0
            - optimizer: str = 'adam'
            - h_units: Optional[List[int]] = None
            - epochs: int = 1000
            - batch_size: int = 32
            - validation_split: float = 0.2
            - callbacks: Optional[List[Callback]] = None
            - verbose: bool = False
            - num_ground: Optional[int] = None
            - monotonicities: str = 'group'
            - errors: str = 'raise'

        :return:
            A tuple containing the `HandlersFactory` as first item, and a list of `Callback` objects as second one.
        """
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
        ds = HandlersFactory(manager=PuzzlesManager(filepath=f'{self.res_folder}/puzzles.csv', **data_args),
                             master_kind='regression',
                             mt_metrics=[MSE(name='loss'), R2(name='metric')],
                             loss='mse',
                             output_act=None,
                             h_units=list(h_units),
                             num_augmented=num_augmented,
                             num_random=num_random,
                             **kwargs)
        return ds, cb

    def restaurants(self,
                    data_args: Optional[dict] = None,
                    h_units: tuple = (128, 128),
                    num_col: int = 1,
                    callbacks: Optional[List[str]] = None,
                    **kwargs) -> Tuple[HandlersFactory, List[Callback]]:
        """Builds an handlers factory for the restaurants dataset.

        :param data_args:
            Custom arguments passed to a `RestaurantsManager` instance.

            - full_grid: bool = False
            - grid_augmented: int = 10
            - grid_ground: Optional[int] = None
            - x_scaling: str = 'std'

        :param h_units:
            The neural network/learner list of hidden units (ignored for TFL).

        :param num_col:
            Number of columns in the final subplot.

        :param callbacks:
            List of callbacks aliases.

        :param kwargs:
            Custom arguments passed to a `HandlersFactory` instance.

            - dataset: Optional[str] = None
            - wandb_project: Optional[str] = 'moving_targets'
            - wandb_entity: Optional[str] = 'giuluck'
            - seed: int = 0
            - optimizer: str = 'adam'
            - h_units: Optional[List[int]] = None
            - epochs: int = 1000
            - batch_size: int = 32
            - validation_split: float = 0.2
            - callbacks: Optional[List[Callback]] = None
            - verbose: bool = False
            - num_augmented: Optional[int] = None
            - num_random: int = 0
            - num_ground: Optional[int] = None
            - monotonicities: str = 'group'
            - errors: str = 'raise'

        :return:
            A tuple containing the `HandlersFactory` as first item, and a list of `Callback` objects as second one.
        """
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
        ds = HandlersFactory(
            manager=RestaurantsManager(**{} if data_args is None else data_args),
            master_kind='regression',
            mt_metrics=[CrossEntropy(name='loss'), AUC(name='metric')],
            loss='binary_crossentropy',
            output_act='sigmoid',
            h_units=list(h_units),
            **kwargs
        )
        return ds, cb

    def default(self,
                data_args: Optional[dict] = None,
                h_units: tuple = (128, 128),
                num_col: int = 1,
                callbacks: Optional[List[str]] = None,
                **kwargs) -> Tuple[HandlersFactory, List[Callback]]:
        """Builds an handlers factory for the default dataset.

        :param data_args:
            Custom arguments passed to a `DefaultManager` instance.

            - full_features: bool = False
            - full_grid: bool = False
            - grid_augmented: int = 10
            - grid_ground: Optional[int] = None
            - x_scaling: str = 'std'
            - train_fraction: float = 0.8

        :param h_units:
            The neural network/learner list of hidden units (ignored for TFL).

        :param num_col:
            Number of columns in the final subplot.

        :param callbacks:
            List of callbacks aliases.

        :param kwargs:
            Custom arguments passed to a `HandlersFactory` instance.

            - dataset: Optional[str] = None
            - wandb_project: Optional[str] = 'moving_targets'
            - wandb_entity: Optional[str] = 'giuluck'
            - seed: int = 0
            - optimizer: str = 'adam'
            - h_units: Optional[List[int]] = None
            - epochs: int = 1000
            - batch_size: int = 32
            - validation_split: float = 0.2
            - callbacks: Optional[List[Callback]] = None
            - verbose: bool = False
            - num_augmented: Optional[int] = None
            - num_random: int = 0
            - num_ground: Optional[int] = None
            - monotonicities: str = 'group'
            - errors: str = 'raise'

        :return:
            A tuple containing the `HandlersFactory` as first item, and a list of `Callback` objects as second one.
        """
        cb, callbacks = self._get_shared_callbacks(callbacks)
        if 'adjustments' in callbacks:
            cb.append(DefaultAdjustments(num_columns=num_col, sorting_attribute='payment',
                                         file_signature=f'{self.temp_folder}/default_adjustments'))
        ds = HandlersFactory(
            manager=DefaultManager(filepath=f'{self.res_folder}/default.csv', **{} if data_args is None else data_args),
            master_kind='regression',
            mt_metrics=[CrossEntropy(name='loss'), Accuracy(name='metric')],
            loss='binary_crossentropy',
            output_act='sigmoid',
            h_units=list(h_units),
            **kwargs
        )
        return ds, cb

    def law(self,
            data_args: Optional[dict] = None,
            h_units: tuple = (128, 128),
            num_col: int = 1,
            callbacks: Optional[List[str]] = None,
            **kwargs) -> Tuple[HandlersFactory, List[Callback]]:
        """Builds an handlers factory for the law dataset.

        :param data_args:
            Custom arguments passed to a `LawManager` instance.

            - full_features: bool = False
            - full_grid: bool = False
            - grid_augmented: int = 8
            - grid_ground: Optional[int] = None
            - x_scaling: Methods = 'std'
            - train_fraction: float = 0.8

        :param h_units:
            The neural network/learner list of hidden units (ignored for TFL).

        :param num_col:
            Number of columns in the final subplot.

        :param callbacks:
            List of callbacks aliases.

        :param kwargs:
            Custom arguments passed to a `HandlersFactory` instance.

            - dataset: Optional[str] = None
            - wandb_project: Optional[str] = 'moving_targets'
            - wandb_entity: Optional[str] = 'giuluck'
            - seed: int = 0
            - optimizer: str = 'adam'
            - h_units: Optional[List[int]] = None
            - epochs: int = 1000
            - batch_size: int = 32
            - validation_split: float = 0.2
            - callbacks: Optional[List[Callback]] = None
            - verbose: bool = False
            - num_augmented: Optional[int] = None
            - num_random: int = 0
            - num_ground: Optional[int] = None
            - monotonicities: str = 'group'
            - errors: str = 'raise'

        :return:
            A tuple containing the `HandlersFactory` as first item, and a list of `Callback` objects as second one.
        """
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
        ds = HandlersFactory(
            manager=LawManager(filepath=f'{self.res_folder}/law.csv', **{} if data_args is None else data_args),
            master_kind='regression',
            mt_metrics=[CrossEntropy(name='loss'), Accuracy(name='metric')],
            loss='binary_crossentropy',
            output_act='sigmoid',
            h_units=list(h_units),
            **kwargs
        )
        return ds, cb
