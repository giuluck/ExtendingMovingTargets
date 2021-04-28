import numpy as np

from moving_targets.metrics.metric import Metric


class ClassFrequenciesStd(Metric):
    def __init__(self, classes=None, name='class frequencies std'):
        super(ClassFrequenciesStd, self).__init__(name=name)
        self.classes = np.arange(classes) if isinstance(classes, int) else classes

    def __call__(self, x, y, p):
        # bincount is similar to np.unique(..., return_counts=True) but allows to fix a minimum number of classes
        # in this way, if the predictions are all the same, the counts will be [n, 0, ..., 0] instead of [n]
        minlength = 1 + max(np.max(self.classes), np.max(y), np.max(p))
        classes_counts = np.bincount(p, minlength=minlength)
        classes_counts = classes_counts if self.classes is None else classes_counts[self.classes]
        return np.std(classes_counts / len(p))


class MonotonicViolation(Metric):
    aggregations = ['average', 'percentage', 'feasible']

    def __init__(self, monotonicities, aggregation='average', eps=1e-3, name='monotonic violation'):
        assert aggregation in self.aggregations, f'aggregation should be in {self.aggregations}'
        super(MonotonicViolation, self).__init__(name=name)
        self.higher_indices = np.array([hi for hi, _ in monotonicities])
        self.lower_indices = np.array([li for _, li in monotonicities])
        self.aggregation = aggregation
        self.eps = eps

    def __call__(self, x, y, p):
        violations = np.array([0]) if len(self.higher_indices) == 0 else p[self.lower_indices] - p[self.higher_indices]
        violations[violations < self.eps] = 0.0
        if self.aggregation == 'average':
            return np.mean(violations)
        elif self.aggregation == 'percentage':
            return np.mean(violations > 0)
        elif self.aggregation == 'feasible':
            return int(np.all(violations <= 0))
        else:
            raise ValueError(f'{self.aggregation} is not a supported violation aggregator')
