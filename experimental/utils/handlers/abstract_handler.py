"""Model Manager."""
import random
import re
import time
from typing import Dict, Any, Union, List, Optional, Callable

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb

from moving_targets.util.typing import Dataset
from src.datasets import AbstractManager
from src.util.preprocessing import Scalers

YInfo = Union[np.ndarray, pd.Series, pd.DataFrame]
"""Data type for the output class or output info."""


def setup(seed: int = 0):
    """Sets the simulation up.

    :param seed:
        The random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def default_config(handler, **config_kwargs) -> Dict:
    """Returns the default configuration of a given model handler.
    
    This is useful to have a dictionary of all the handler's parameters for logging information even if they are not
    directly passed to the handler itself, as default values are used.

    :param handler:
        The model handler.

    :param config_kwargs:
        Custom config arguments that are added to default ones.
        
    :return:
        A dictionary containing the default model handler configuration, plus the custom arguments.
    """
    config = dict()
    for k, v in handler.__dict__.items():
        if k in ['manager', 'wandb_args', 'wandb_config']:
            pass
        elif isinstance(v, dict):
            config.update({f"{k.replace('_args', '')}/{kk}": vv for kk, vv in v.items()})
        else:
            config[k] = v
    config.update(config_kwargs)
    return config


class Fold:
    """Data class containing the information of a fold for k-fold cross-validation."""

    def __init__(self, x, y: YInfo, scalers: Scalers, validation: Dataset):
        """
        :param x:
            The input data.

        :param y:
            The output data/info.

        :param scalers:
            The tuple of x/y scalers.

        :param validation:
            A shared validation dataset which is common among all the k folds.
        """
        self.x = x
        """The input data."""

        self.y: YInfo = y
        """The output data/info."""

        self.scalers: Scalers = scalers
        """The tuple of x/y scalers."""

        self.validation: Dataset = validation
        """A shared validation dataset which is common among all the k folds."""


class AbstractHandler:
    """Abstract model handler."""

    def __init__(self,
                 manager: AbstractManager,
                 model: Optional[str] = None,
                 dataset: Optional[str] = None,
                 wandb_name: Optional[str] = None,
                 wandb_project: Optional[str] = 'moving_targets',
                 wandb_entity: Optional[str] = 'giuluck',
                 wandb_config: Callable = default_config,
                 seed: int = 0):
        """
        :param manager:
            The dataset manager.

        :param model:
            The machine learning model name.

        :param dataset:
            The dataset name.

        :param wandb_name:
            The Weights&Biases run name. If None, no Weights&Biases instance is created.

        :param wandb_project:
            The Weights&Biases project name. If wandb_name is None, this is ignored.

        :param wandb_entity:
            The Weights&Biases entity name. If wandb_name is None, this is ignored.

        :param wandb_config:
            The Weights&Biases configuration function which is in charge of returning the configuration dictionary.

        :param seed:
            The random seed.
        """

        self.manager = manager
        """The dataset manager."""

        self.dataset: Optional[str] = dataset
        """The dataset name."""

        self.model: str = self.__class__.__name__.replace('Handler', '') if model is None else model
        """The machine learning model name."""

        self.seed: int = seed
        """The random seed."""

        self.wandb_config: Callable = wandb_config
        """The Weights&Biases configuration function which is in charge of returning the configuration dictionary."""

        self.wandb_args: Optional[Dict] = None if wandb_name is None else dict(
            name=wandb_name,
            project=wandb_project,
            entity=wandb_entity
        )
        """The Weights&Biases instance arguments."""

        # if no explicit dataset name is passed, this is retrieved from the dataset manager class name
        if dataset is None:
            # class name, split by capital letters
            # first empty string (classes begin with capitals) and final 'Test' string are removed
            # the result is joined with spaces and lower-cased
            self.dataset = ' '.join(re.split('(?=[A-Z])', self.manager.__class__.__name__)[1:-1]).lower()

    def fit(self, fold: Fold) -> Any:
        """Fits the machine learning model on the given fold.

        :param fold:
            A fold for k-fold cross-validation.

        :return:
            The machine learning model itself.
        """
        raise NotImplementedError("Please implement method 'fit'")

    def get_folds(self, num_folds: Optional[int]) -> Union[Fold, List[Fold]]:
        """Builds the k folds for k-fold cross-validation.

        :param num_folds:
            The number of folds. If None, train/test split only is performed.

        :return:
            A list of folds (or a single fold if num_folds is None).
        """
        folds: List[Fold] = []
        if num_folds is None:
            data, scalers = self.manager.get_folds(num_folds=None)
            x, y = data['train']
            return Fold(x=x, y=y, scalers=scalers, validation=data)
        else:
            for data, scalers in self.manager.get_folds(num_folds=num_folds):
                x, y = data['train']
                fold = Fold(x=x, y=y, scalers=scalers, validation=data)
                folds.append(fold)
        return folds

    def validate(self,
                 num_folds: int = 10,
                 folds_index: Optional[List[int]] = None,
                 summary_args: Optional[Dict] = None):
        """Runs a k-fold cross-validation experiment.

        :param num_folds:
            The number of folds. If None, train/test split only is performed.

        :param folds_index:
            The list of folds index to be validated. If None, all of them are validated.

        :param summary_args:
            The dictionary of arguments for the evaluation summary.
        """
        for i, fold in enumerate(self.get_folds(num_folds=num_folds)):
            if folds_index is None or i in folds_index:
                self._run_instance(fold=fold, index=i, summary_args=summary_args)

    def test(self, summary_args: Optional[Dict] = None):
        """Runs a train/test experiment.

        :param summary_args:
            The dictionary of arguments for the evaluation summary.
        """
        fold = self.get_folds(num_folds=None)
        self._run_instance(fold=fold, index='test', summary_args=summary_args)

    def _run_instance(self, fold: Fold, index: Union[int, str], summary_args: Dict):
        """Runs an experiment instance.

        :param fold:
            A fold for k-fold cross-validation.

        :param index:
            The fold index.

        :param summary_args:
            The dictionary of arguments for the summary plot. If None, no evaluation summary is called.
        """
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
