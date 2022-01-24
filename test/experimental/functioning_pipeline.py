"""Pipelines Functioning Tests."""
import unittest
from typing import Optional, Dict

from test.experimental.abstract_pipeline import TestPipelines, EXPECTED_TESTS

unittest.TestLoader.sortTestMethodsUsing = lambda self, t1, t2: EXPECTED_TESTS.index(t1) - EXPECTED_TESTS.index(t2)


class TestFunctioning(TestPipelines):
    def _grid_ground(self) -> Optional[int]:
        # Use minimal number of augmented grid points.
        return 2

    def _model_parameters(self, model: str, dataset: str) -> Optional[Dict]:
        # Builds a dictionary with minimal parameters as we are testing the program functioning, not its correctness.
        if model in ['mlp', 'tfl', 'sbr univariate']:
            return dict(epochs=0)
        elif model == 'sbr':
            return dict(epochs=0, num_ground=2, num_random=0)
        elif model == 'mt':
            return dict(lrn_epochs=0, aug_num_ground=2, aug_num_random=0, mt_iterations=1, mst_master_kind='regression')
        else:
            ValueError(f"unsupported model '{model}'")

    def _test_dataset_model(self, dataset: str, model: str, **kwargs):
        # Runs tests for partial/full features and for augmented/full grid, i.e:
        # > basic pair of tests for each dataset: explicit full grid and grid created with data augmentation
        tests_args = [dict(full_grid=True), dict(full_grid=False, grid_ground=self._grid_ground())]
        # > 'synthetic', 'restaurants' and 'univariate' datasets do not accept 'full_features' parameter, the others do
        if dataset not in ['synthetic', 'restaurants', 'cars univariate']:
            tests_args = [dict(full_features=False, **data_args) for data_args in tests_args]
            # > 'cars univariate' only deals with partial features (i.e., the single attribute)
            # > the others can deal with full features, but full grid is not used due to computational limits
            if dataset != 'cars univariate':
                tests_args.append(dict(full_features=True, full_grid=False, grid_ground=self._grid_ground()))
        # at the end, run all tests
        for data_args in tests_args:
            self._test(dataset, model, data_args={**data_args, **kwargs})
