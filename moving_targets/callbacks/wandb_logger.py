import wandb

from moving_targets.callbacks.logger import Logger


class WandBLogger(Logger):
    instance = wandb

    def __init__(self, project, entity, run_name, **kwargs):
        super(WandBLogger, self).__init__()
        self.config = dict(project=project, entity=entity, name=run_name, config=kwargs)

    def on_process_start(self, macs, x, y, val_data):
        wandb.init(**self.config)

    def on_iteration_end(self, macs, x, y, val_data, iteration):
        wandb.log({k: self.cache[k] for k in sorted(self.cache)})
        self.cache = {}

    def on_process_end(self, macs, x, y, val_data):
        wandb.finish()
