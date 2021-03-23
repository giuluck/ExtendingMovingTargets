from typing import List

from src.moving_targets.learners import Learner
from src.moving_targets.masters import Master
from src.moving_targets.metrics import Metric, MultiMetric
from src.moving_targets.loggers import Logger, MultiLogger


class MACS:
    def __init__(self, learner: Learner, master: Master, metrics: List[Metric] = None,
                 alpha=1., beta=1., initial_step='pretraining', use_prob=False):
        super(MACS, self).__init__()
        assert alpha > 0, "alpha should be a positive number"
        assert beta > 0, "beta should be a positive number"
        assert initial_step in ['pretraining', 'projection'], "initial step should be 'pretraining' or 'projection'"
        self.learner = learner
        self.master = master
        self.alpha = alpha
        self.beta = beta
        self.initial_step = initial_step
        self.use_prob = use_prob
        self.metrics = MultiMetric([]) if metrics is None else MultiMetric(metrics)

    def fit(self, x, y, iterations=1, val_data=None, loggers: List[Logger] = None):
        # check user input
        assert iterations > 0, "there should be at least one iteration"
        val_x, val_y = (None, None) if val_data is None else val_data
        loggers = MultiLogger([]) if loggers is None else MultiLogger(loggers)

        start_iteration = 0
        loggers.on_process_start(self)
        if self.initial_step == 'pretraining':
            loggers.on_pretraining_start(self)
            loggers.on_training_start(self)
            self.learner.fit(x, y)
            loggers.on_training_end(self, x, y, val_x, val_y)
            loggers.on_pretraining_end(self)
        elif self.initial_step == 'projection':
            start_iteration = 1  # the first step is performed outside the for loop, thus iterations will start from 1
            loggers.on_iteration_start(self, start_iteration)
            self._step(x, y, y.reshape(-1), val_x, val_y, alpha=1e6, beta=self.beta, use_prob=False, loggers=loggers)
            loggers.on_iteration_end(self, start_iteration)

        # algorithm core
        for i in range(start_iteration, iterations):
            loggers.on_iteration_start(self, i + 1)
            pred = self.learner.predict_proba(x) if self.use_prob else self.learner.predict(x)
            self._step(x, y, pred, val_x, val_y, self.alpha, self.beta, self.use_prob, loggers)
            loggers.on_iteration_end(self, i + 1)
        loggers.on_process_end(self)

    def predict(self, x):
        return self.learner.predict(x)

    def predict_proba(self, x):
        return self.learner.predict_proba(x)

    def evaluate(self, x, y):
        pred = self.predict(x)
        return self.metrics(x, y, pred)

    def _step(self, x, y, pred, val_x, val_y, alpha, beta, use_prob, loggers):
        loggers.on_adjustment_start(self)
        adj_y = self.master.adjust_targets(y, pred, alpha, beta, use_prob)
        loggers.on_adjustment_end(self, x, y, adj_y)
        loggers.on_training_start(self)
        self.learner.fit(x, adj_y)
        loggers.on_training_end(self, x, y, val_x, val_y)
