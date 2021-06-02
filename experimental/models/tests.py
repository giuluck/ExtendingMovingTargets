"""Testing Script."""

from experimental.datasets.managers import PuzzlesTest
from experimental.models.managers import TFLManager

if __name__ == '__main__':
    manager = TFLManager(test_manager=PuzzlesTest(), epochs=1000, wandb_name=None)
    manager.test(summary_args={})
