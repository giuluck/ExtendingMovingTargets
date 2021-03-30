import wandb

from src.moving_targets.callbacks import Logger


class WandBLogger(Logger):
    def __init__(self, project, entity, run_name, **kwargs):
        super(WandBLogger, self).__init__()
        self.config = dict(project=project, entity=entity, name=run_name, config=kwargs)

    def on_process_start(self, macs):
        wandb.init(**self.config)

    def on_iteration_end(self, macs, idx):
        wandb.log({k: self.cache[k] for k in sorted(self.cache)})
        self.cache = {}

    def on_process_end(self, macs):
        wandb.finish()
