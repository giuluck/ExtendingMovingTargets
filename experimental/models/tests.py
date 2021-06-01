"""Testing Script."""

from experimental.models.managers import SBRManager, MLPManager
from experimental.datasets.managers import CarsTest, DefaultTest

if __name__ == '__main__':
    manager = SBRManager(test_manager=DefaultTest(), verbose=False, wandb_name=None)
    manager.validate(num_folds=10, summary_args={})
