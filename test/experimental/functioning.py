"""Pipelines Functioning Tests."""
from typing import Optional, Dict

from test.experimental.pipelines import TestPipelines


class TestFunctioning(TestPipelines):
    def _model_parameters(self, model: str, dataset: str) -> Optional[Dict]:
        if model in ['mlp', 'sbr', 'sbr univariate', 'tfl']:
            return dict(epochs=0)
        elif model == 'mt':
            return dict(lrn_epochs=0, aug_num_ground=2, mt_iterations=1)
        else:
            ValueError(f"unsupported model '{model}'")

    def _summary_args(self, model: str, dataset: str) -> Optional[Dict]:
        return None
