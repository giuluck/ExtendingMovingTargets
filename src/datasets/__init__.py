"""Dataset managers."""

from src.datasets.abstract_manager import AbstractManager
from src.datasets.fairness_managers import AdultManager, CommunitiesManager
from src.datasets.balanced_managers import DotaManager, IrisManager, RedwineManager, ShuttleManager, WhitewineManager
