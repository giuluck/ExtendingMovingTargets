"""Dataset managers."""

from src.managers.abstract_manager import AbstractManager
from src.managers.balanced_managers import DotaManager, IrisManager, RedwineManager, ShuttleManager, WhitewineManager
from src.managers.fairness_managers import AdultManager, CommunitiesManager


def get_manager(dataset: str, **kwargs) -> AbstractManager:
    """Returns an `AbstractManager` instance given the dataset name.

    :param dataset:
        The dataset name.

    :param kwargs:
        Any additional implementation-dependent arguments for the `AbstractManager`.

    :return:
        The `AbstractManager` instance related to the given dataset.

    :raise `ValueError`:
        If the dataset is not known.
    """
    if dataset == 'iris':
        return IrisManager(**kwargs)
    elif dataset == 'redwine':
        return RedwineManager(**kwargs)
    elif dataset == 'whitewine':
        return WhitewineManager(**kwargs)
    elif dataset == 'shuttle':
        return ShuttleManager(**kwargs)
    elif dataset == 'dota':
        return DotaManager(**kwargs)
    elif dataset == 'adult':
        return AdultManager(**kwargs)
    elif dataset == 'communities':
        return CommunitiesManager(**kwargs)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")
