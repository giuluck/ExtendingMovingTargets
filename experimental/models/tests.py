"""Testing Script."""

from experimental.models.managers import SBRManager
from experimental.datasets.managers import CarsTest

if __name__ == '__main__':
    manager = SBRManager(test_manager=CarsTest(), wandb_name=None)
    manager.test(summary_args={})
