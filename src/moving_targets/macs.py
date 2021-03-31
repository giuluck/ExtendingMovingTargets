from typing import List

from src.moving_targets.learners import Learner
from src.moving_targets.masters import Master
from src.moving_targets.metrics import Metric
from src.moving_targets.callbacks import Logger, Callback, FileLogger


class MACS(Logger):
    def __init__(self, learner: Learner, master: Master, init_step: str = 'pretraining', metrics: List[Metric] = None):
        super(MACS, self).__init__()
        assert init_step in ['pretraining', 'projection'], "initial step should be 'pretraining' or 'projection'"
        self.learner = learner
        self.master = master
        self.init_step = init_step
        self.metrics = [] if metrics is None else metrics

    def fit(self, x, y, iterations=1, val_data=None, callbacks: List[Callback] = 'stdout'):
        # check user input
        assert iterations > 0, "there should be at least one iteration"
        val_data = {} if val_data is None else (val_data if isinstance(val_data, dict) else {'val': val_data})

        # handle callbacks
        callbacks = [FileLogger()] if callbacks == 'stdout' else ([] if callbacks is None else callbacks)
        self._update_callbacks(callbacks, lambda c: c.on_process_start(self, x, y, val_data))

        # handle pretraining
        if self.init_step == 'pretraining':
            self._update_callbacks(callbacks, lambda c: c.on_pretraining_start(self, x, y, val_data))
            self._update_callbacks(callbacks, lambda c: c.on_training_start(self, x, y, val_data))
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self.learner.fit(self, x, y, iteration=-1)
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self._update_callbacks(callbacks, lambda c: c.on_training_end(self, x, y, val_data))
            self._update_callbacks(callbacks, lambda c: c.on_pretraining_end(self, x, y, val_data))

        # algorithm core
        for iteration in range(iterations):
            self._update_callbacks(callbacks, lambda c: c.on_iteration_start(self, x, y, val_data, iteration))
            self._update_callbacks(callbacks, lambda c: c.on_adjustment_start(self, x, y, val_data))
            # ---------------------------------------------- MASTER  STEP ----------------------------------------------
            outputs = self.master.adjust_targets(self, x, y, iteration)
            if not isinstance(outputs, dict):
                outputs = {'y': outputs}
            # ---------------------------------------------- MASTER  STEP ----------------------------------------------
            self._update_callbacks(callbacks, lambda c: c.on_adjustment_end(self, x, y, outputs['y'], val_data))
            self._update_callbacks(callbacks, lambda c: c.on_training_start(self, x, y, val_data))
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self.learner.fit(self, x, iteration=iteration, **outputs)
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self._update_callbacks(callbacks, lambda c: c.on_training_end(self, x, y, val_data))
            self._update_callbacks(callbacks, lambda c: c.on_iteration_end(self, x, y, val_data, iteration))
        self._update_callbacks(callbacks, lambda c: c.on_process_end(self, x, y, val_data))

    def predict(self, x):
        return self.learner.predict(x)

    def evaluate(self, x, y):
        return {metric.name: metric(x, y, self.predict(x)) for metric in self.metrics}

    def on_pretraining_end(self, macs, x, y, val_data):
        self.on_iteration_end(macs, x, y, val_data, 'pretraining')

    def on_iteration_end(self, macs, x, y, val_data, iteration):
        logs = {'iteration': iteration}
        # log metrics on training data
        for metric in self.metrics:
            logs[metric.name] = metric(x, y, self.predict(x))
        # log metrics on validation data
        for name, (xx, yy) in val_data.items():
            pp = self.predict(xx)
            for metric in self.metrics:
                logs[f'{name}_{metric.name}'] = metric(xx, yy, pp)
        self.log(**logs)

    def _update_callbacks(self, callbacks, routine):
        routine(self)                         # run callback routine for moving targets object itself
        for callback in callbacks:            #
            if isinstance(callback, Logger):  # update cache for loggers only
                callback.log(**self.cache)    #
            routine(callback)                 # run callback routine for each external callback
        self.cache = {}                       # eventually clear the cache
