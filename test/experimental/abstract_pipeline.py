"""Pipelines Abstract Tests."""
import unittest
from typing import Dict, Tuple, Type, Callable, Optional, List

import numpy as np

from experimental.utils import DatasetFactory
from experimental.utils.handlers import ClassificationFactory, RegressionFactory, MLPHandler, SBRHandler, \
    UnivariateSBRHandler, TFLHandler, MTHandler
from src.datasets import CarsManager, SyntheticManager, PuzzlesManager, RestaurantsManager, DefaultManager, LawManager

RES_FOLDER: str = '../../res/'
"""The resource folder where the datasets are placed."""

DATASETS: Dict[str, Tuple[Type, Type]] = {
    'cars univariate': (RegressionFactory, CarsManager),
    'cars': (RegressionFactory, CarsManager),
    'synthetic': (RegressionFactory, SyntheticManager),
    'puzzles': (RegressionFactory, PuzzlesManager),
    'restaurants': (ClassificationFactory, RestaurantsManager),
    'default': (ClassificationFactory, DefaultManager),
    'law': (ClassificationFactory, LawManager)
}
"""A dictionary which links each dataset (str) to a tuple of TypeClasses representing the kind of factory (either
`RegressionFactory` or `ClassificationFactory`) and the kind of manager (which is dataset-dependent)."""

MODELS: Dict[str, Tuple[Callable, Type]] = {
    'mlp': (lambda f, p: f.get_mlp(**p), MLPHandler),
    'sbr': (lambda f, p: f.get_sbr(**p), SBRHandler),
    'sbr univariate': (lambda f, p: f.get_univariate_sbr(**p), UnivariateSBRHandler),
    'tfl': (lambda f, p: f.get_tfl(**p), TFLHandler),
    'mt': (lambda f, p: f.get_mt(**p), MTHandler)
}
"""A dictionary which links each model (str) to a tuple made of a Callable function which is used to obtain the model
given a factory f and a dictionary of parameters p, and a TypeClass representing the kind of model handler."""

EXPECTED_TESTS: List[str] = ['test_expected_tests'] + [
    f'test_{dataset}_{model}' for model, dataset in zip(*np.array(np.meshgrid(
        ['mlp', 'sbr', 'tfl', 'mt'],
        ['univariate_cars', 'cars', 'synthetic', 'puzzles', 'restaurants', 'default', 'law']
    )).reshape(2, -1))
]
"""An (ordered) list of the tests that are expected, namely each combination of dataset and model."""

unittest.TestLoader.sortTestMethodsUsing = lambda self, x, y: EXPECTED_TESTS.index(x) - EXPECTED_TESTS.index(y)


class TestPipelines(unittest.TestCase):
    """Abstract class used as template method class for the correctness and functioning tests."""

    def _grid_ground(self) -> Optional[int]:
        """Defines the number of grid samples for each for each attribute to evaluate the functioning of the algorithms.
        If None, no grid evaluation is performed.

        :returns:
            The number of grid samples.
        """
        pass

    def _model_parameters(self, model: str, dataset: str) -> Optional[Dict]:
        """Links each model and dataset to the given dictionary of model parameters depending on the tests.

        :param model:
            The model name.

        :param dataset:
            The dataset name.

        :returns:
            Either None or a dictionary of model parameters.
        """
        pass

    def _summary_args(self, model: str, dataset: str, **kwargs) -> Optional[Dict]:
        """Links each model and dataset to the given dictionary of summary parameters depending on the tests.

        :param model:
            The model name.

        :param dataset:
            The dataset name.

        :param kwargs:
            Additional information.

        :returns:
            Either None or a dictionary of summary parameters.
        """
        pass

    def _test(self, dataset: str, model: str, **kwargs):
        """Core of the test class which is in charge of checking the assertions.

        :param dataset:
            The name of the dataset.

        :param model:
            The name of the model.

        :param kwargs:
            Additional data arguments which are passed to the ".get_dataset()" function.
        """
        expected_factory, expected_manager = DATASETS[dataset]
        handler_routine, expected_handler = MODELS[model]
        try:
            # we can now use these information to retrieve the correct pipeline
            factory, _ = DatasetFactory(res_folder=RES_FOLDER).get_dataset(name=dataset, **kwargs)
            manager = factory.manager
            handler = handler_routine(factory, self._model_parameters(model, dataset) or {})
            test_routine = handler.test
            # variables are placed before so to avoid warnings due to incorrect type assignment after assertIsInstance()
            self.assertIsInstance(factory, expected_factory)
            self.assertIsInstance(manager, expected_manager)
            self.assertIsInstance(handler, expected_handler)
            test_routine(summary_args=self._summary_args(model, dataset, **kwargs))
        except Exception as exception:
            self.fail(f'{exception}\nArguments: {kwargs}')

    def _test_dataset_model(self, dataset: str, model: str, **kwargs):
        """Runs multiple "self._test()" instances depending on the dataset, the model, and the additional arguments.

        :param dataset:
            The name of the dataset.

        :param model:
            The name of the model.

        :param kwargs:
            Additional data arguments.
        """
        raise NotImplementedError("Please implement method '_test_dataset_model'")

    def test_expected_tests(self):
        """This is a meta test which is used to guarantee that all the tests have been correctly implemented, i.e.,
        that all the combinations of the seven datasets and the four models have the correct name."""
        actual_tests = [t for t in dir(self) if t.startswith('test_')]
        self.assertSetEqual(set(actual_tests), set(EXPECTED_TESTS))

    def test_univariate_cars_mlp(self):
        """Dataset: 'Cars Univariate', Model: 'MLP'."""
        self._test_dataset_model('cars univariate', 'mlp')

    def test_univariate_cars_sbr(self):
        """Dataset: 'Cars Univariate', Model: 'SBR Univariate'."""
        self._test_dataset_model('cars univariate', 'sbr univariate')

    def test_univariate_cars_tfl(self):
        """Dataset: 'Cars Univariate', Model: 'TFL'."""
        self._test_dataset_model('cars univariate', 'tfl')

    def test_univariate_cars_mt(self):
        """Dataset: 'Cars Univariate', Model: 'MT'."""
        self._test_dataset_model('cars univariate', 'mt')

    def test_cars_mlp(self):
        """Dataset: 'Cars', Model: 'MLP'."""
        self._test_dataset_model('cars', 'mlp')

    def test_cars_sbr(self):
        """Dataset: 'Cars', Model: 'SBR'."""
        self._test_dataset_model('cars', 'sbr')

    def test_cars_tfl(self):
        """Dataset: 'Cars', Model: 'TFL'."""
        self._test_dataset_model('cars', 'tfl')

    def test_cars_mt(self):
        """Dataset: 'Cars', Model: 'MT'."""
        self._test_dataset_model('cars', 'mt')

    def test_synthetic_mlp(self):
        """Dataset: 'Synthetic', Model: 'MLP'."""
        self._test_dataset_model('synthetic', 'mlp')

    def test_synthetic_sbr(self):
        """Dataset: 'Synthetic', Model: 'SBR'."""
        self._test_dataset_model('synthetic', 'sbr')

    def test_synthetic_tfl(self):
        """Dataset: 'Synthetic', Model: 'TFL'."""
        self._test_dataset_model('synthetic', 'tfl')

    def test_synthetic_mt(self):
        """Dataset: 'Synthetic', Model: 'MT'."""
        self._test_dataset_model('synthetic', 'mt')

    def test_puzzles_mlp(self):
        """Dataset: 'Puzzles', Model: 'MLP'."""
        self._test_dataset_model('puzzles', 'mlp')

    def test_puzzles_sbr(self):
        """Dataset: 'Puzzles', Model: 'SBR'."""
        self._test_dataset_model('puzzles', 'sbr')

    def test_puzzles_tfl(self):
        """Dataset: 'Puzzles', Model: 'TFL'."""
        self._test_dataset_model('puzzles', 'tfl')

    def test_puzzles_mt(self):
        """Dataset: 'Puzzles', Model: 'MT'."""
        self._test_dataset_model('puzzles', 'mt')

    def test_restaurants_mlp(self):
        """Dataset: 'Restaurants', Model: 'MLP'."""
        self._test_dataset_model('restaurants', 'mlp')

    def test_restaurants_sbr(self):
        """Dataset: 'Restaurants', Model: 'SBR'."""
        self._test_dataset_model('restaurants', 'sbr')

    def test_restaurants_tfl(self):
        """Dataset: 'Restaurants', Model: 'TFL'."""
        self._test_dataset_model('restaurants', 'tfl')

    def test_restaurants_mt(self):
        """Dataset: 'Restaurants', Model: 'MT'."""
        self._test_dataset_model('restaurants', 'mt')

    def test_default_mlp(self):
        """Dataset: 'Default', Model: 'MLP'."""
        self._test_dataset_model('default', 'mlp', train_fraction=0.025)

    def test_default_sbr(self):
        """Dataset: 'Default', Model: 'SBR'."""
        self._test_dataset_model('default', 'sbr', train_fraction=0.025)

    def test_default_tfl(self):
        """Dataset: 'Default', Model: 'TFL'."""
        self._test_dataset_model('default', 'tfl', train_fraction=0.025)

    def test_default_mt(self):
        """Dataset: 'Default', Model: 'MT'."""
        self._test_dataset_model('default', 'mt', train_fraction=0.025)

    def test_law_mlp(self):
        """Dataset: 'Law', Model: 'MLP'."""
        self._test_dataset_model('law', 'mlp', train_fraction=0.03)

    def test_law_sbr(self):
        """Dataset: 'Law', Model: 'SBR'."""
        self._test_dataset_model('law', 'sbr', train_fraction=0.03)

    def test_law_tfl(self):
        """Dataset: 'Law', Model: 'TFL'."""
        self._test_dataset_model('law', 'tfl', train_fraction=0.03)

    def test_law_mt(self):
        """Dataset: 'Law', Model: 'MT'."""
        self._test_dataset_model('law', 'mt', train_fraction=0.03)
