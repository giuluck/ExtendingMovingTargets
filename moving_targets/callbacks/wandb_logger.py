"""Weights&Biases Callback"""

from typing import Dict, Optional

import wandb

from moving_targets.callbacks.logger import Logger
from moving_targets.util.typing import Matrix, Vector, Dataset, Iteration


class WandBLogger(Logger):
    """Logs the training information on a Weights&Biases instance."""

    instance = wandb
    """The Weights&Biases instance."""

    def __init__(self, project: str, entity: str, run_name: str, **wandb_kwargs):
        """
        :param project:
            Weights&Biases project name.

        :param entity:
            Weights&Biases entity name.

        :param run_name:
            Weights&Biases run name.

        :param wandb_kwargs:
            Weights&Biases run configuration.
        """
        super(WandBLogger, self).__init__()

        self._project: str = project
        """The Weights&Biases project name."""

        self._entity: str = entity
        """The Weights&Biases entity name."""

        self._run_name: str = run_name
        """The Weights&Biases run name."""

        self.config: Dict = wandb_kwargs
        """The Weights&Biases run configuration."""

    def on_process_start(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], **additional_kwargs):
        wandb.init(project=self._project, entity=self._entity, name=self._run_name, config=self.config)

    def on_iteration_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration,
                         **additional_kwargs):
        wandb.log({k: self._cache[k] for k in sorted(self._cache)})
        self._cache = {}

    def on_process_end(self, macs, val_data: Optional[Dataset], **additional_kwargs):
        wandb.finish()
