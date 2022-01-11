"""Dataset managers."""

from src.managers.abstract_manager import AbstractManager, Fold, Config
from src.managers.balanced_managers import DotaManager, IrisManager, RedwineManager, ShuttleManager, WhitewineManager
from src.managers.fairness_managers import AdultManager, CommunitiesManager


def get_manager(dataset: str, **config) -> AbstractManager:
    """Returns an `AbstractManager` instance given the dataset name.

    :param dataset:
        The dataset name.

    :param config:
        Custom`arguments for a Config instance.

    :return:
        The `AbstractManager` instance related to the given dataset.

    :raise `ValueError`:
        If the dataset is not known.
    """
    config = Config(**config)
    if dataset == 'iris':
        return IrisManager(config=config)
    elif dataset == 'redwine':
        return RedwineManager(config=config)
    elif dataset == 'whitewine':
        return WhitewineManager(config=config)
    elif dataset == 'shuttle':
        return ShuttleManager(config=config)
    elif dataset == 'dota':
        return DotaManager(config=config)
    elif dataset == 'adult':
        return AdultManager(config=config)
    elif dataset == 'communities':
        return CommunitiesManager(config=config)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")
