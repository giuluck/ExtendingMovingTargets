from typing import List

from src.moving_targets.metrics import Metric


class MultiMetric(Metric):
    def __init__(self, metrics: List[Metric]):
        super(MultiMetric, self).__init__(name=None)
        self.metrics = metrics

    def __call__(self, x, y, pred):
        return {metric.name: metric(x, y, pred) for metric in self.metrics}
