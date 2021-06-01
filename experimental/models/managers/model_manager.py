"""Model Manager."""
import time

import wandb
from typing import Dict, Any, Union, List, Optional

from src.util.dictionaries import merge_dictionaries
from experimental.datasets.managers import TestManager, Fold


# noinspection PyMissingOrEmptyDocstring
class ModelManager:
    def __init__(self,
                 test_manager: TestManager,
                 wandb_name: Optional[str] = None,
                 project: str = 'shape_constraints',
                 entity: str = 'giuluck',
                 **kwargs):
        self.test_manager: TestManager = test_manager
        # WANDB INFO
        self.project: str = project
        self.entity: str = entity
        self.wandb_name: str = wandb_name
        self.model_name: str = self.__class__.__name__.replace('Manager', '')
        self.config: Dict = {'model': self.model_name, 'dataset': test_manager.name}
        # FIT INFO
        self.output_act = test_manager.learner_args['output_act']
        self.h_units = test_manager.learner_args['h_units']
        self.loss = test_manager.learner_args['loss']
        self.optimizer = test_manager.learner_args['optimizer']
        self.fit_info: Dict[str, Any] = kwargs

    def fit(self, fold: Fold) -> Any:
        raise NotImplementedError("Please implement method 'fit'")

    def get_folds(self, num_folds: int, extrapolation: bool, compute_monotonicities: bool = True) -> List[Fold]:
        return self.test_manager.get_folds(num_folds=num_folds,
                                           extrapolation=extrapolation,
                                           compute_monotonicities=compute_monotonicities)

    def validate(self, num_folds: int = 10, summary_args: Dict = None):
        for i, fold in enumerate(self.get_folds(num_folds=num_folds, extrapolation=False)):
            self._run_instance(fold=fold, index=i, summary_args=summary_args)

    def test(self, extrapolation: bool = False, summary_args: Dict = None):
        fold = self.get_folds(num_folds=1, extrapolation=extrapolation)[0]
        self._run_instance(fold=fold, index='test', summary_args=summary_args)

    def _run_instance(self, fold: Fold, index: Union[int, str], summary_args: Dict):
        TestManager.setup(seed=self.test_manager.seed)
        start_time = time.time()
        model = self.fit(fold=fold)
        elapsed_time = time.time() - start_time
        if self.wandb_name is not None:
            config = {**self.config, 'fold': index}
            wandb.init(project=self.project, entity=self.entity, name=self.wandb_name, config=config)
            metrics = self.test_manager.dataset.metrics_summary(model, return_type='dict', **fold.validation)
            violations = self.test_manager.dataset.violations_summary(model, return_type='dict')
            wandb.log({**violations, **{f'{t}_metric': v for t, v in metrics.items()}, 'elapsed_time': elapsed_time})
            wandb.finish()
        if summary_args is not None:
            summary_args = merge_dictionaries(self.test_manager.summary_args, summary_args)
            self.test_manager.dataset.evaluation_summary(model, **fold.validation, **summary_args)
