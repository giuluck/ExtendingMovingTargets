"""Pipelines Functioning Tests."""
from typing import Optional, Dict

from test.experimental.pipelines import TestPipelines


class TestFunctioning(TestPipelines):
    def _grid_ground(self) -> Optional[int]:
        return 2

    def _model_parameters(self, model: str, dataset: str) -> Optional[Dict]:
        if model in ['mlp', 'tfl', 'sbr univariate']:
            return dict(epochs=0)
        elif model == 'sbr':
            return dict(epochs=0, num_ground=2, num_random=0)
        elif model == 'mt':
            return dict(lrn_epochs=0, aug_num_ground=2, aug_num_random=0, mt_iterations=1)
        else:
            ValueError(f"unsupported model '{model}'")

    def _summary_args(self, model: str, dataset: str) -> Optional[Dict]:
        return None
