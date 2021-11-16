"""Pipelines Correctness Tests (manual inspection of plots needed)."""
from typing import Optional, Dict

from test.experimental.abstract_pipeline import TestPipelines


class TestCorrectness(TestPipelines):
    def _grid_ground(self) -> Optional[int]:
        # Use minimal number of augmented grid points.
        return 2

    def _model_parameters(self, model: str, dataset: str) -> Optional[Dict]:
        # Builds a dictionary with custom parameters for MT and an empty dictionary for the other models.
        if model in ['mlp', 'sbr', 'sbr univariate', 'tfl']:
            return dict()
        elif model == 'mt':
            return dict(mt_iterations=1, mst_backend='cplex', mst_master_kind='regression',
                        lrn_loss='binary_crossentropy' if dataset in ['restaurants', 'default', 'law'] else 'mse')
        else:
            ValueError(f"unsupported model '{model}' with dataset '{dataset}'")

    def _summary_args(self, model: str, dataset: str, **kwargs) -> Optional[Dict]:
        # Links each model and dataset to the given dictionary of summary parameters depending on the tests.
        model_name = model.upper()
        model_name += ' - '
        model_name += ' '.join([s.capitalize() for s in dataset.split(' ')])
        model_name += ' - '
        model_name += 'Full Grid' if kwargs['data_args']['full_grid'] else 'Augmented Grid'
        return dict(model_name=model_name)

    def _test_dataset_model(self, dataset: str, model: str, **kwargs):
        # Runs tests for partial features (which are the only one supporting plots), i.e.:
        # > basic pair of tests for each dataset: explicit full grid and grid created with data augmentation
        tests_args = [dict(full_grid=True), dict(full_grid=False, grid_ground=self._grid_ground())]
        # > 'synthetic' and 'restaurants' datasets do not accept 'full_features' parameter, the others do
        if dataset not in ['synthetic', 'restaurants']:
            tests_args = [dict(full_features=False, **data_args) for data_args in tests_args]
        # at the end, run all tests
        for data_args in tests_args:
            self._test(dataset, model, data_args={**data_args, **kwargs})
