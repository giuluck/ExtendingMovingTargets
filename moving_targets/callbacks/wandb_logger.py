"""Weights&Biases Callback"""

from typing import Dict, Optional as Opt

import wandb

from moving_targets.callbacks.logger import Logger
from moving_targets.util.typing import Matrix, Vector, Dataset, Iteration


class WandBLogger(Logger):
    """Logs the training information on a Weights&Biases instance."""
    instance = wandb

    def __init__(self, project: str, entity: str, run_name: str, **kwargs):
        super(WandBLogger, self).__init__()
        self.config: Dict = dict(project=project, entity=entity, name=run_name, config=kwargs)

    # noinspection PyMissingOrEmptyDocstring
    def on_process_start(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], **kwargs):
        wandb.init(**self.config)

    # noinspection PyMissingOrEmptyDocstring
    def on_iteration_end(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        wandb.log({k: self.cache[k] for k in sorted(self.cache)})
        self.cache = {}

    # noinspection PyMissingOrEmptyDocstring
    def on_process_end(self, macs, val_data: Opt[Dataset], **kwargs):
        wandb.finish()
