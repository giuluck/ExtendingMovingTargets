from typing import List

from src.moving_targets.learners import Learner
from src.moving_targets.masters import Master
from src.moving_targets.metrics import Metric
from src.moving_targets.callbacks import Logger, Callback


class MACS(Logger):
    def __init__(self, learner: Learner, master: Master, init_step: str = 'pretraining', metrics: List[Metric] = None):
        super(MACS, self).__init__()
        assert init_step in ['pretraining', 'projection'], "initial step should be 'pretraining' or 'projection'"
        self.learner = learner
        self.master = master
        self.init_step = init_step
        self.metrics = [] if metrics is None else metrics

    def fit(self, x, y, iterations=1, callbacks: List[Callback] = None):
        # check user input
        assert iterations > 0, "there should be at least one iteration"

        # handle callbacks
        callbacks = [] if callbacks is None else callbacks
        self._update_callbacks(callbacks, lambda c: c.on_process_start(self))

        # handle pretraining
        if self.init_step == 'pretraining':
            self._update_callbacks(callbacks, lambda c: c.on_pretraining_start(self))
            self._update_callbacks(callbacks, lambda c: c.on_training_start(self))
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self.learner.fit(self, x, y, iteration=-1)
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self._update_callbacks(callbacks, lambda c: c.on_training_end(self, x, y))
            self._update_callbacks(callbacks, lambda c: c.on_pretraining_end(self))

        # algorithm core
        for iteration in range(iterations):
            self._update_callbacks(callbacks, lambda c: c.on_iteration_start(self, iteration))
            self._update_callbacks(callbacks, lambda c: c.on_adjustment_start(self))
            # ---------------------------------------------- MASTER  STEP ----------------------------------------------
            outputs = self.master.adjust_targets(self, x, y, iteration)
            if not isinstance(outputs, dict):
                outputs = {'y': outputs}
            # ---------------------------------------------- MASTER  STEP ----------------------------------------------
            self._update_callbacks(callbacks, lambda c: c.on_adjustment_end(self, x, y, adj_y=outputs['y']))
            self._update_callbacks(callbacks, lambda c: c.on_training_start(self))
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self.learner.fit(self, x, iteration=iteration, **outputs)
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self._update_callbacks(callbacks, lambda c: c.on_training_end(self, x, y))
            self._update_callbacks(callbacks, lambda c: c.on_iteration_end(self, iteration))
        self._update_callbacks(callbacks, lambda c: c.on_process_end(self))

    def predict(self, x):
        return self.learner.predict(x)

    def evaluate(self, x, y):
        return {metric.name: metric(x, y, self.predict(x)) for metric in self.metrics}

    def _update_callbacks(self, callbacks, routine):
        routine(self)                         # run callback routine for moving targets object itself
        for callback in callbacks:            #
            routine(callback)                 # run callback routine for each external callback
            if isinstance(callback, Logger):  #
                callback.log(**self.cache)    # update loggers and eventually clear the cache
        self.cache = {}
