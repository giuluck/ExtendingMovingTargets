import time
import wandb


class Logger:
    def __init__(self):
        super(Logger, self).__init__()

    def on_process_start(self, macs):
        pass

    def on_process_end(self, macs):
        pass

    def on_pretraining_start(self, macs):
        pass

    def on_pretraining_end(self, macs):
        pass

    def on_iteration_start(self, macs, idx):
        pass

    def on_iteration_end(self, macs, idx):
        pass

    def on_training_start(self, macs):
        pass

    def on_training_end(self, macs, x, y, val_x, val_y):
        train_scores = macs.metrics(x, y, macs.predict(x))
        val_scores = None
        if val_x is not None and val_y is not None:
            val_scores = macs.metrics(val_x, val_y, macs.predict(val_x))
        return train_scores, val_scores

    def on_adjustment_start(self, macs):
        pass

    def on_adjustment_end(self, macs, x, y, adj_y):
        return macs.metrics(x, y, adj_y)


class MultiLogger(Logger):
    def __init__(self, loggers):
        super(MultiLogger, self).__init__()
        self.loggers = loggers

    def _update_loggers(self, routine):
        for log in self.loggers:
            routine(log)

    def on_process_start(self, macs):
        self._update_loggers(lambda log: log.on_process_start(macs))

    def on_process_end(self, macs):
        self._update_loggers(lambda log: log.on_process_end(macs))

    def on_pretraining_start(self, macs):
        self._update_loggers(lambda log: log.on_pretraining_start(macs))

    def on_pretraining_end(self, macs):
        self._update_loggers(lambda log: log.on_pretraining_end(macs))

    def on_iteration_start(self, macs, idx):
        self._update_loggers(lambda log: log.on_iteration_start(macs, idx))

    def on_iteration_end(self, macs, idx):
        self._update_loggers(lambda log: log.on_iteration_end(macs, idx))

    def on_training_start(self, macs):
        self._update_loggers(lambda log: log.on_training_start(macs))

    def on_training_end(self, macs, x, y, x_val, y_val):
        self._update_loggers(lambda log: log.on_training_end(macs, x, y, x_val, y_val))

    def on_adjustment_start(self, macs):
        self._update_loggers(lambda log: log.on_adjustment_start(macs))

    def on_adjustment_end(self, macs, x, y, adj_y):
        self._update_loggers(lambda log: log.on_adjustment_end(macs, x, y, adj_y))


class WandBLogger(Logger):
    def __init__(self, project, entity, run_name, **kwargs):
        super(WandBLogger, self).__init__()
        self.config = dict(project=project, entity=entity, name=run_name)
        self.logs = kwargs

    def on_process_start(self, macs):
        super(WandBLogger, self).on_process_start(macs)
        wandb.init(**self.config)

    def on_process_end(self, macs):
        super(WandBLogger, self).on_process_end(macs)
        wandb.finish()

    def on_pretraining_start(self, macs):
        super(WandBLogger, self).on_pretraining_start(macs)
        self.on_iteration_start(macs, 0)

    def on_pretraining_end(self, macs):
        super(WandBLogger, self).on_pretraining_end(macs)
        self.on_iteration_end(macs, 0)

    def on_iteration_start(self, macs, idx):
        super(WandBLogger, self).on_iteration_start(idx, macs)
        self.logs['iteration'] = idx
        self.logs['time/iteration'] = time.time()

    def on_iteration_end(self, macs, idx):
        super(WandBLogger, self).on_iteration_end(macs, idx)
        self.logs['time/iteration'] = time.time() - self.logs['time/iteration']
        wandb.log(self.logs)
        self.logs = {}

    def on_training_start(self, macs):
        super(WandBLogger, self).on_training_start(macs)
        self.logs['time/learner'] = time.time()

    def on_training_end(self, macs, x, y, x_val, y_val):
        train_scores, val_scores = super(WandBLogger, self).on_training_end(macs, x, y, x_val, y_val)
        self.logs['time/learner'] = time.time() - self.logs['time/learner']
        self._update_logs(train_scores, 'learner-train')
        if val_scores is not None:
            self._update_logs(val_scores, 'learner-val')

    def on_adjustment_start(self, macs):
        super(WandBLogger, self).on_adjustment_start(macs)
        self.logs['time/master'] = time.time()

    def on_adjustment_end(self, macs, x, y, adj_y):
        master_scores = super(WandBLogger, self).on_adjustment_end(macs, x, y, adj_y)
        self.logs['time/master'] = time.time() - self.logs['time/master']
        self._update_logs(master_scores, 'master')

    def _update_logs(self, logs, name):
        for key, value in logs.items():
            self.logs[f'{name}/{key}'] = value
