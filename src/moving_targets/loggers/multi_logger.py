from typing import List

from src.moving_targets.loggers import Logger


class MultiLogger(Logger):
    def __init__(self, loggers: List[Logger]):
        super(MultiLogger, self).__init__()
        self.loggers = loggers

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

    def _update_loggers(self, routine):
        for log in self.loggers:
            routine(log)
