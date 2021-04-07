import time

from moving_targets.callbacks import Logger, FileLogger
from moving_targets.callbacks.history import History


class MACS(Logger):
    def __init__(self, learner, master, init_step='pretraining', metrics=None):
        super(MACS, self).__init__()
        assert init_step in ['pretraining', 'projection'], "initial step should be 'pretraining' or 'projection'"
        self.learner = learner
        self.master = master
        self.init_step = init_step
        self.metrics = [] if metrics is None else metrics
        self.history = History()
        self.time = None

    def fit(self, x, y, iterations=1, val_data=None, callbacks=None, verbose=True):
        # check user input
        assert iterations > 0, "there should be at least one iteration"
        val_data = {} if val_data is None else (val_data if isinstance(val_data, dict) else {'val': val_data})

        # handle callbacks and verbosity
        callbacks = [] if callbacks is None else callbacks
        if verbose:
            callbacks = [FileLogger()] + callbacks
        self._update_callbacks(callbacks, lambda c: c.on_process_start(self, x, y, val_data))

        # handle pretraining
        if self.init_step == 'pretraining':
            self._update_callbacks(callbacks, lambda c: c.on_pretraining_start(self, x, y, val_data))
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self.learner.fit(self, x, y, iteration='pretraining')
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self._update_callbacks(callbacks, lambda c: c.on_pretraining_end(self, x, y, val_data))

        # algorithm core
        for iteration in range(1, iterations + 1):
            self._update_callbacks(callbacks, lambda c: c.on_iteration_start(self, x, y, val_data, iteration))
            self._update_callbacks(callbacks, lambda c: c.on_adjustment_start(self, x, y, val_data, iteration))
            # ---------------------------------------------- MASTER  STEP ----------------------------------------------
            adj = self.master.adjust_targets(self, x, y, iteration)
            if not isinstance(adj, dict):
                adj = {'y': adj}
            # ---------------------------------------------- MASTER  STEP ----------------------------------------------
            self._update_callbacks(callbacks, lambda c: c.on_adjustment_end(self, x, y, adj['y'], val_data, iteration))
            self._update_callbacks(callbacks, lambda c: c.on_training_start(self, x, y, val_data, iteration))
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self.learner.fit(self, x, iteration=iteration, **adj)
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self._update_callbacks(callbacks, lambda c: c.on_training_end(self, x, y, val_data, iteration))
            self._update_callbacks(callbacks, lambda c: c.on_iteration_end(self, x, y, val_data, iteration))
        self._update_callbacks(callbacks, lambda c: c.on_process_end(self, x, y, val_data))

        return self.history

    def predict(self, x):
        return self.learner.predict(x)

    def evaluate(self, x, y):
        return {metric.name: metric(x, y, self.predict(x)) for metric in self.metrics}

    def on_iteration_start(self, macs, x, y, val_data, iteration):
        self.time = time.time()

    def on_iteration_end(self, macs, x, y, val_data, iteration):
        logs = {'iteration': iteration, 'elapsed time': time.time() - self.time}
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
        # run callback routine for macs object itself and history object
        routine(self)
        # run callback routine for the history logger and for each external callback
        for callback in [self.history] + callbacks:
            # if the callback is a logger, log the internal cache before calling the routine (and eventually clear it)
            if isinstance(callback, Logger):
                callback.log(**self.cache)
            routine(callback)
        self.cache = {}
