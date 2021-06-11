"""Model Manager."""
import random
import re
import time
from typing import Dict, Any, Union, List, Optional, Callable

import numpy as np
import tensorflow as tf
import wandb
from pandas import DataFrame

from moving_targets.util.typing import Vector, Matrix, Dataset
from src.datasets import AbstractManager
from src.util.preprocessing import Scalers

YInfo = Union[Vector, DataFrame]


# noinspection PyMissingOrEmptyDocstring
def setup(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# noinspection PyMissingOrEmptyDocstring
def default_config(handler, **kwargs) -> Dict:
    config = dict()
    for k, v in handler.__dict__.items():
        if k in ['manager', 'wandb_args', 'wandb_config']:
            pass
        elif isinstance(v, dict):
            config.update({f"{k.replace('_args', '')}/{kk}": vv for kk, vv in v.items()})
        else:
            config[k] = v
    config.update(kwargs)
    return config


# noinspection PyMissingOrEmptyDocstring
class Fold:
    def __init__(self, x: Matrix, y: YInfo, scalers: Scalers, validation: Dataset):
        self.x: Matrix = x
        self.y: YInfo = y
        self.scalers: Scalers = scalers
        self.validation: Dataset = validation


# noinspection PyMissingOrEmptyDocstring
class AbstractHandler:
    def __init__(self,
                 manager: AbstractManager,
                 model: Optional[str] = None,
                 dataset: Optional[str] = None,
                 wandb_name: Optional[str] = None,
                 wandb_project: Optional[str] = 'shape_constraints',
                 wandb_entity: Optional[str] = 'giuluck',
                 wandb_config: Callable = default_config,
                 seed: int = 0):
        self.manager = manager
        # dataset & model info
        if dataset is None:
            # class name, split by capital letters
            # first empty string (classes begin with capitals) and final 'Test' string are removed
            # the result is joined with spaces and lower-cased
            self.dataset: str = ' '.join(re.split('(?=[A-Z])', self.manager.__class__.__name__)[1:-1]).lower()
        else:
            self.dataset: str = dataset
        self.model: str = self.__class__.__name__.replace('Handler', '') if model is None else model
        self.seed: int = seed
        self.wandb_config: Callable = wandb_config
        self.wandb_args: Optional[Dict] = None if wandb_name is None else dict(
            name=wandb_name,
            project=wandb_project,
            entity=wandb_entity
        )

    def fit(self, fold: Fold) -> Any:
        raise NotImplementedError("Please implement method 'fit'")

    def get_folds(self, num_folds: int, extrapolation: bool) -> List[Fold]:
        folds: List[Fold] = []
        for data, scalers in self.manager.load_data(num_folds=num_folds, extrapolation=extrapolation):
            x, y = data['train']
            fold = Fold(x=x, y=y, scalers=scalers, validation=data)
            folds.append(fold)
        return folds

    def validate(self, num_folds: int = 10, folds_index: Optional[List[int]] = None,
                 extrapolation: bool = False, summary_args: Optional[Dict] = None):
        for i, fold in enumerate(self.get_folds(num_folds=num_folds, extrapolation=extrapolation)):
            if folds_index is None or i in folds_index:
                self._run_instance(fold=fold, index=i, summary_args=summary_args)

    def test(self, extrapolation: bool = False, summary_args: Optional[Dict] = None):
        fold = self.get_folds(num_folds=1, extrapolation=extrapolation)[0]
        self._run_instance(fold=fold, index='test', summary_args=summary_args)

    def _run_instance(self, fold: Fold, index: Union[int, str], summary_args: Dict):
        setup(seed=self.seed)
        start_time = time.time()
        model = self.fit(fold=fold)
        elapsed_time = time.time() - start_time
        if self.wandb_args is not None:
            wandb.init(**self.wandb_args, config=self.wandb_config(self, fold=index))
            losses = self.manager.losses_summary(model, return_type='dict', **fold.validation)
            metrics = self.manager.metrics_summary(model, return_type='dict', **fold.validation)
            violations = self.manager.violations_summary(model, return_type='dict')
            wandb.log({
                **violations,
                **{f'{t}_loss': v for t, v in losses.items()},
                **{f'{t}_metric': v for t, v in metrics.items()},
                'elapsed_time': elapsed_time
            })
            wandb.finish()
        if summary_args is not None:
            self.manager.evaluation_summary(model, **fold.validation, **summary_args)
