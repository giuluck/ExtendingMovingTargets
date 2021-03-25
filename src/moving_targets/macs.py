from typing import List

from src.moving_targets.learners import Learner
from src.moving_targets.masters import Master
from src.moving_targets.metrics import Metric, MultiMetric
from src.moving_targets.loggers import Logger, MultiLogger


class MACS:
    def __init__(self, learner: Learner, master: Master, init_step: str = 'pretraining', metrics: List[Metric] = None):
        super(MACS, self).__init__()
        assert init_step in ['pretraining', 'projection'], "initial step should be 'pretraining' or 'projection'"
        self.learner = learner
        self.master = master
        self.init_step = init_step
        self.metrics = MultiMetric([]) if metrics is None else MultiMetric(metrics)

    def fit(self, x, y, iterations=1, val_data=None, loggers: List[Logger] = None):
        # check user input
        assert iterations > 0, "there should be at least one iteration"
        val_x, val_y = (None, None) if val_data is None else val_data
        loggers = MultiLogger([]) if loggers is None else MultiLogger(loggers)

        loggers.on_process_start(self)
        if self.init_step == 'pretraining':
            loggers.on_pretraining_start(self)
            self._learner_step(x, y, val_x, val_y, iteration=-1, loggers=loggers)
            loggers.on_pretraining_end(self)

        # algorithm core
        for iteration in range(iterations):
            loggers.on_iteration_start(self, iteration)
            adj_y = self._master_step(x, y, iteration, loggers)
            self._learner_step(x, adj_y, val_x, val_y, iteration, loggers)
            loggers.on_iteration_end(self, iteration)
        loggers.on_process_end(self)

    def predict(self, x):
        return self.learner.predict(x)

    def predict_proba(self, x):
        return self.learner.predict_proba(x)

    def evaluate(self, x, y):
        return self.metrics(x, y, self.predict(x))

    def _learner_step(self, x, y, val_x, val_y, iteration, loggers):
        loggers.on_training_start(self)
        self.learner.fit(self, x, y, iteration)
        loggers.on_training_end(self, x, y, val_x, val_y)

    def _master_step(self, x, y, iteration, loggers):
        loggers.on_adjustment_start(self)
        adj_y = self.master.adjust_targets(self, x, y, iteration)
        loggers.on_adjustment_end(self, x, y, adj_y)
        return adj_y
