"""Weights&Biases Callback"""

from typing import Dict, Optional as Opt

import wandb

from moving_targets.callbacks.logger import Logger
from moving_targets.util.typing import Matrix, Vector, Dataset, Iteration


class WandBLogger(Logger):
    """Logs the training information on a Weights&Biases instance.

    Args:
        project: Weights&Biases project name.
        entity: Weights&Biases entity name.
        run_name: Weights&Biases run name.
        **kwargs: Weights&Biases run configuration.
    """
    instance = wandb

    def __init__(self, project: str, entity: str, run_name: str, **kwargs):
        super(WandBLogger, self).__init__()
        self.project: str = project
        self.entity: str = entity
        self.run_name: str = run_name
        self.config: Dict = kwargs

    # noinspection PyMissingOrEmptyDocstring
    def on_process_start(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], **kwargs):
        wandb.init(project=self.project, entity=self.entity, name=self.run_name, config=self.config)

    # noinspection PyMissingOrEmptyDocstring
    def on_iteration_end(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        wandb.log({k: self.cache[k] for k in sorted(self.cache)})
        self.cache = {}

    # noinspection PyMissingOrEmptyDocstring
    def on_process_end(self, macs, val_data: Opt[Dataset], **kwargs):
        wandb.finish()
