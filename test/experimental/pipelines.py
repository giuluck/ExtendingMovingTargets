"""Pipelines Abstract Tests."""
import inspect
import unittest
from typing import Dict, Tuple, Type, Callable, Optional

import numpy as np

from experimental.utils import DatasetFactory
from experimental.utils.handlers import ClassificationFactory, RegressionFactory, MLPHandler, SBRHandler, \
    UnivariateSBRHandler, TFLHandler, MTHandler
from src.datasets import CarsManager, SyntheticManager, PuzzlesManager, RestaurantsManager, DefaultManager, LawManager

RES_FOLDER: str = '../../res/'
DATASETS: Dict[str, Tuple[Type, Type]] = {
    'cars univariate': (RegressionFactory, CarsManager),
    'cars': (RegressionFactory, CarsManager),
    'synthetic': (RegressionFactory, SyntheticManager),
    'puzzles': (RegressionFactory, PuzzlesManager),
    'restaurants': (ClassificationFactory, RestaurantsManager),
    'default': (ClassificationFactory, DefaultManager),
    'law': (ClassificationFactory, LawManager)
}
MODELS: Dict[str, Tuple[Callable, Type]] = {
    'mlp': (lambda f, p: f.get_mlp(**p), MLPHandler),
    'sbr': (lambda f, p: f.get_sbr(**p), SBRHandler),
    'sbr univariate': (lambda f, p: f.get_univariate_sbr(**p), UnivariateSBRHandler),
    'tfl': (lambda f, p: f.get_tfl(**p), TFLHandler),
    'mt': (lambda f, p: f.get_mt(**p), MTHandler)
}


# noinspection DuplicatedCode
class TestPipelines(unittest.TestCase):
    def _grid_ground(self) -> Optional[int]:
        raise NotImplementedError("please implement method '_grid_ground()'")

    def _model_parameters(self, model: str, dataset: str) -> Optional[Dict]:
        raise NotImplementedError("please implement method '_model_parameters()'")

    def _summary_args(self, model: str, dataset: str) -> Optional[Dict]:
        raise NotImplementedError("please implement method '_summary_args()'")

    def _test(self, **kwargs):
        # the caller function name is in the form 'test_<optional: univariate>_<dataset>_<model>'
        # thus we split by '_' to retrieve the dataset and the model as the last two splits, then deal with univariate
        caller_function_name: str = inspect.stack()[1][3]
        caller_info = caller_function_name.split('_')
        dataset, model = caller_info[-2], caller_info[-1]
        if 'univariate' in caller_function_name:
            dataset, model = f'{dataset} univariate', 'sbr univariate' if model == 'sbr' else model
        expected_factory, expected_manager = DATASETS[dataset]
        handler_routine, expected_handler = MODELS[model]
        # noinspection PyBroadException
        try:
            # we can now use these information to retrieve the correct pipeline
            factory, _ = DatasetFactory(res_folder=RES_FOLDER).get_dataset(name=dataset, **kwargs)
            self.assertIsInstance(factory, expected_factory)
            # noinspection PyUnresolvedReferences
            self.assertIsInstance(factory.manager, expected_manager)
            handler = handler_routine(factory, self._model_parameters(model, dataset) or {})
            self.assertIsInstance(handler, expected_handler)
            # noinspection PyUnresolvedReferences
            handler.test(summary_args=self._summary_args(model, dataset))
        except Exception:
            self.fail(f'Arguments: {kwargs}')

    def test_meta_tests(self):
        datasets, models = np.meshgrid(
            ['univariate_cars', 'cars', 'synthetic', 'puzzles', 'restaurants', 'default', 'law'],
            ['mlp', 'sbr', 'tfl', 'mt']
        )
        expected_tests = [f'test_{d}_{m}' for d, m in zip(datasets.flatten(), models.flatten())]
        actual_tests = [t for t in dir(self) if t.startswith('test_') and t != 'test_meta_tests']
        self.assertSetEqual(set(actual_tests), set(expected_tests))

    def test_univariate_cars_mlp(self):
        self._test(data_args=dict(full_features=False, full_grid=True))
        self._test(data_args=dict(full_features=False, full_grid=False, grid_ground=self._grid_ground()))

    def test_univariate_cars_sbr(self):
        self._test(data_args=dict(full_features=False, full_grid=True))
        self._test(data_args=dict(full_features=False, full_grid=False, grid_ground=self._grid_ground()))

    def test_univariate_cars_tfl(self):
        self._test(data_args=dict(full_features=False, full_grid=True))
        self._test(data_args=dict(full_features=False, full_grid=False, grid_ground=self._grid_ground()))

    def test_univariate_cars_mt(self):
        self._test(data_args=dict(full_features=False, full_grid=True))
        self._test(data_args=dict(full_features=False, full_grid=False, grid_ground=self._grid_ground()))

    def test_cars_mlp(self):
        self._test(data_args=dict(full_features=False, full_grid=True))
        self._test(data_args=dict(full_features=False, full_grid=False, grid_ground=self._grid_ground()))
        self._test(data_args=dict(full_features=True, full_grid=False, grid_ground=self._grid_ground()))

    def test_cars_sbr(self):
        self._test(data_args=dict(full_features=False, full_grid=True))
        self._test(data_args=dict(full_features=False, full_grid=False, grid_ground=self._grid_ground()))
        self._test(data_args=dict(full_features=True, full_grid=False, grid_ground=self._grid_ground()))

    def test_cars_tfl(self):
        self._test(data_args=dict(full_features=False, full_grid=True))
        self._test(data_args=dict(full_features=False, full_grid=False, grid_ground=self._grid_ground()))
        self._test(data_args=dict(full_features=True, full_grid=False, grid_ground=self._grid_ground()))

    def test_cars_mt(self):
        self._test(data_args=dict(full_features=False, full_grid=True))
        self._test(data_args=dict(full_features=False, full_grid=False, grid_ground=self._grid_ground()))
        self._test(data_args=dict(full_features=True, full_grid=False, grid_ground=self._grid_ground()))

    def test_synthetic_mlp(self):
        self._test(data_args=dict(full_grid=True))
        self._test(data_args=dict(full_grid=False, grid_ground=self._grid_ground()))

    def test_synthetic_sbr(self):
        self._test(data_args=dict(full_grid=True))
        self._test(data_args=dict(full_grid=False, grid_ground=self._grid_ground()))

    def test_synthetic_tfl(self):
        self._test(data_args=dict(full_grid=True))
        self._test(data_args=dict(full_grid=False, grid_ground=self._grid_ground()))

    def test_synthetic_mt(self):
        self._test(data_args=dict(full_grid=True))
        self._test(data_args=dict(full_grid=False, grid_ground=self._grid_ground()))

    def test_puzzles_mlp(self):
        self._test(data_args=dict(full_features=False, full_grid=True))
        self._test(data_args=dict(full_features=False, full_grid=False, grid_ground=self._grid_ground()))
        self._test(data_args=dict(full_features=True, full_grid=False, grid_ground=self._grid_ground()))

    def test_puzzles_sbr(self):
        self._test(data_args=dict(full_features=False, full_grid=True))
        self._test(data_args=dict(full_features=False, full_grid=False, grid_ground=self._grid_ground()))
        self._test(data_args=dict(full_features=True, full_grid=False, grid_ground=self._grid_ground()))

    def test_puzzles_tfl(self):
        self._test(data_args=dict(full_features=False, full_grid=True))
        self._test(data_args=dict(full_features=False, full_grid=False, grid_ground=self._grid_ground()))
        self._test(data_args=dict(full_features=True, full_grid=False, grid_ground=self._grid_ground()))

    def test_puzzles_mt(self):
        self._test(data_args=dict(full_features=False, full_grid=True))
        self._test(data_args=dict(full_features=False, full_grid=False, grid_ground=self._grid_ground()))
        self._test(data_args=dict(full_features=True, full_grid=False, grid_ground=self._grid_ground()))

    def test_restaurants_mlp(self):
        self._test(data_args=dict(full_grid=True))
        self._test(data_args=dict(full_grid=False, grid_ground=self._grid_ground()))

    def test_restaurants_sbr(self):
        self._test(data_args=dict(full_grid=True))
        self._test(data_args=dict(full_grid=False, grid_ground=self._grid_ground()))

    def test_restaurants_tfl(self):
        self._test(data_args=dict(full_grid=True))
        self._test(data_args=dict(full_grid=False, grid_ground=self._grid_ground()))

    def test_restaurants_mt(self):
        self._test(data_args=dict(full_grid=True))
        self._test(data_args=dict(full_grid=False, grid_ground=self._grid_ground()))

    def test_default_mlp(self):
        ground = self._grid_ground()
        self._test(data_args=dict(full_features=False, full_grid=True))
        self._test(data_args=dict(full_features=False, full_grid=False, train_fraction=0.025, grid_ground=ground))
        self._test(data_args=dict(full_features=True, full_grid=False, train_fraction=0.025, grid_ground=ground))

    def test_default_sbr(self):
        ground = self._grid_ground()
        self._test(data_args=dict(full_features=False, full_grid=True))
        self._test(data_args=dict(full_features=False, full_grid=False, train_fraction=0.025, grid_ground=ground))
        self._test(data_args=dict(full_features=True, full_grid=False, train_fraction=0.025, grid_ground=ground))

    def test_default_tfl(self):
        ground = self._grid_ground()
        self._test(data_args=dict(full_features=False, full_grid=True))
        self._test(data_args=dict(full_features=False, full_grid=False, train_fraction=0.025, grid_ground=ground))
        self._test(data_args=dict(full_features=True, full_grid=False, train_fraction=0.025, grid_ground=ground))

    def test_default_mt(self):
        ground = self._grid_ground()
        self._test(data_args=dict(full_features=False, full_grid=True))
        self._test(data_args=dict(full_features=False, full_grid=False, train_fraction=0.025, grid_ground=ground))
        self._test(data_args=dict(full_features=True, full_grid=False, train_fraction=0.025, grid_ground=ground))

    def test_law_mlp(self):
        ground = self._grid_ground()
        self._test(data_args=dict(full_features=False, full_grid=True))
        self._test(data_args=dict(full_features=False, full_grid=False, train_fraction=0.03, grid_ground=ground))
        self._test(data_args=dict(full_features=True, full_grid=False, train_fraction=0.03, grid_ground=ground))

    def test_law_sbr(self):
        ground = self._grid_ground()
        self._test(data_args=dict(full_features=False, full_grid=True))
        self._test(data_args=dict(full_features=False, full_grid=False, train_fraction=0.03, grid_ground=ground))
        self._test(data_args=dict(full_features=True, full_grid=False, train_fraction=0.03, grid_ground=ground))

    def test_law_tfl(self):
        ground = self._grid_ground()
        self._test(data_args=dict(full_features=False, full_grid=True))
        self._test(data_args=dict(full_features=False, full_grid=False, train_fraction=0.03, grid_ground=ground))
        self._test(data_args=dict(full_features=True, full_grid=False, train_fraction=0.03, grid_ground=ground))

    def test_law_mt(self):
        ground = self._grid_ground()
        self._test(data_args=dict(full_features=False, full_grid=True))
        self._test(data_args=dict(full_features=False, full_grid=False, train_fraction=0.03, grid_ground=ground))
        self._test(data_args=dict(full_features=True, full_grid=False, train_fraction=0.03, grid_ground=ground))
