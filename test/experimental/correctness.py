"""Pipelines Correctness Tests (manual inspection)."""
from typing import Optional, Dict

from test.experimental.pipelines import TestPipelines


class TestCorrectness(TestPipelines):
    def _grid_ground(self) -> Optional[int]:
        return 2

    def _model_parameters(self, model: str, dataset: str) -> Optional[Dict]:
        if model in ['mlp', 'sbr', 'sbr univariate', 'tfl']:
            return dict()
        elif model == 'mt' and dataset in ['cars univariate', 'cars', 'synthetic', 'puzzles']:
            return dict(mt_iterations=1)
        elif model == 'mt' and dataset in ['restaurants', 'default', 'law']:
            return dict(mt_iterations=1, mst_master_kind='regression', lrn_loss='mse')
        else:
            ValueError(f"unsupported model '{model}' with dataset '{dataset}'")

    def _summary_args(self, model: str, dataset: str) -> Optional[Dict]:
        return dict(model_name=model.upper())

    def test_cars_mlp(self):
        self._test(data_args=dict(full_features=True, full_grid=False, grid_ground=self._grid_ground()))
