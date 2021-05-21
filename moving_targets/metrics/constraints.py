from typing import Any, List, Tuple

import numpy as np

from moving_targets.metrics.metric import Metric


class ClassFrequenciesStd(Metric):
    def __init__(self, classes: Any = None, name: str = 'class frequencies std'):
        super(ClassFrequenciesStd, self).__init__(name=name)
        self.classes: np.ndarray = np.arange(classes) if isinstance(classes, int) else classes

    def __call__(self, x, y, p) -> float:
        # bincount is similar to np.unique(..., return_counts=True) but allows to fix a minimum number of classes
        # in this way, if the predictions are all the same, the counts will be [n, 0, ..., 0] instead of [n]
        minlength = 1 + max(np.max(self.classes), np.max(y), np.max(p))
        classes_counts = np.bincount(p, minlength=minlength)
        classes_counts = classes_counts if self.classes is None else classes_counts[self.classes]
        classes_counts = classes_counts / len(p)
        return classes_counts.std()


class MonotonicViolation(Metric):
    def __init__(self, monotonicities: List[Tuple[int, int]], aggregation: str = 'average', eps: float = 1e-3,
                 name: str = 'monotonic violation'):
        super(MonotonicViolation, self).__init__(name=name)
        self.higher_indices: np.ndarray = np.array([hi for hi, _ in monotonicities])
        self.lower_indices: np.ndarray = np.array([li for _, li in monotonicities])
        if aggregation == 'average':
            self.aggregate = lambda violations: np.mean(violations)
        elif aggregation == 'percentage':
            self.aggregate = lambda violations: np.mean(violations > 0)
        elif aggregation == 'feasible':
            self.aggregate = lambda violations: int(np.all(violations <= 0))
        else:
            raise ValueError(f"{aggregation} should be in ['average', 'percentage', 'feasible']")
        self.eps = eps

    def __call__(self, x, y, p) -> float:
        violations = np.array([0]) if len(self.higher_indices) == 0 else p[self.lower_indices] - p[self.higher_indices]
        violations[violations < self.eps] = 0.0
        return self.aggregate(violations)
