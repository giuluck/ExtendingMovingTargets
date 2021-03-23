import time
import wandb

from src.moving_targets.loggers import Logger


class WandBLogger(Logger):
    def __init__(self, project, entity, run_name, **kwargs):
        super(WandBLogger, self).__init__()
        self.config = dict(project=project, entity=entity, name=run_name)
        self.logs = kwargs

    def on_process_start(self, macs):
        wandb.init(**self.config)

    def on_process_end(self, macs):
        wandb.finish()

    def on_pretraining_start(self, macs):
        self.on_iteration_start(macs, 0)

    def on_pretraining_end(self, macs):
        self.on_iteration_end(macs, 0)

    def on_iteration_start(self, macs, idx):
        self.logs['iteration'] = idx
        self.logs['time/iteration'] = time.time()

    def on_iteration_end(self, macs, idx):
        self.logs['time/iteration'] = time.time() - self.logs['time/iteration']
        wandb.log({k: self.logs[k] for k in sorted(self.logs)})
        self.logs = {}

    def on_training_start(self, macs):
        self.logs['time/learner'] = time.time()

    def on_training_end(self, macs, x, y, val_x, val_y):
        self.logs['time/learner'] = time.time() - self.logs['time/learner']
        self._update_logs(macs.metrics(x, y, macs.predict(x)), 'learner-train')
        if val_x is not None and val_y is not None:
            self._update_logs(macs.metrics(val_x, val_y, macs.predict(val_x)), 'learner-val')

    def on_adjustment_start(self, macs):
        super(WandBLogger, self).on_adjustment_start(macs)
        self.logs['time/master'] = time.time()

    def on_adjustment_end(self, macs, x, y, adj_y):
        self.logs['time/master'] = time.time() - self.logs['time/master']
        self._update_logs(macs.metrics(x, y, adj_y), 'master')

    def _update_logs(self, logs, name):
        for key, value in logs.items():
            self.logs[f'{name}/{key}'] = value
