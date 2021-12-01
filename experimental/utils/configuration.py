"""Experiment Configuration."""

import random
from typing import Dict

import numpy as np
import tensorflow as tf

from moving_targets.util.typing import Matrix, Dataset, Vector
from src.datasets import AbstractManager, IrisManager, RedwineManager, WhitewineManager, ShuttleManager, DotaManager
from src.util.preprocessing import Scalers, Scaler


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


def get_manager(dataset: str, res_folder: str = '../../res', **manager_kwargs) -> AbstractManager:
    """Returns an `AbstractManager` instance given the dataset name.

    :param dataset:
        The dataset name.

    :param res_folder:
        The resource folder path.

    :param manager_kwargs:
        Any additional implementation-dependent arguments for the `AbstractManager`.

    :return:
        The `AbstractManager` instance related to the given dataset.

    :raise `ValueError`:
        If the dataset is not known.
    """
    res_folder = res_folder.strip('/')
    if dataset == 'iris':
        return IrisManager(filepath=f'{res_folder}/iris.csv', **manager_kwargs)
    elif dataset == 'redwine':
        return RedwineManager(filepath=f'{res_folder}/redwine.csv', **manager_kwargs)
    elif dataset == 'whitewine':
        return WhitewineManager(filepath=f'{res_folder}/whitewine.csv', **manager_kwargs)
    elif dataset == 'shuttle':
        return ShuttleManager(filepath=f'{res_folder}/shuttle.trn', **manager_kwargs)
    elif dataset == 'dota':
        return DotaManager(filepath=f'{res_folder}/dota2.csv', **manager_kwargs)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")


class Fold:
    """Data class containing the information of a fold for k-fold cross-validation."""

    def __init__(self, x: Matrix, y: Vector, scalers: Scalers, validation: Dataset):
        """
        :param x:
            The input data.

        :param y:
            The output data.

        :param scalers:
            The tuple of x/y scalers.

        :param validation:
            A shared validation dataset which is common among all the k folds.
        """
        xsc, ysc = scalers
        xsc = Scaler.get_default(x.shape[-1]) if xsc is None else xsc
        ysc = Scaler.get_default(y.shape[-1]) if ysc is None else ysc

        self.x: Matrix = xsc.fit_transform(x)
        """The input data."""

        self.y: Vector = ysc.fit_transform(y)
        """The output data/info."""

        self.scalers: Scalers = scalers
        """The tuple of x/y scalers."""

        self.validation: Dataset = {k: (xsc.transform(x), ysc.transform(y)) for k, (x, y) in validation.items()}
        """A shared validation dataset which is common among all the k folds."""
