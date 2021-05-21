"""Constraints Metrics."""

import numpy as np
from typing import Optional

from moving_targets.metrics.metric import Metric
from moving_targets.utils.typing import Classes, Monotonicities, Matrix, Vector


class ClassFrequenciesStd(Metric):
    """Standard Deviation of the Class Frequencies, usually constrained to be null.

    Args:
        classes: the number of classes or a vector of class labels.
        name: the name of the metric.
    """

    def __init__(self, classes: Optional[Classes] = None, name: str = 'class frequencies std'):
        super(ClassFrequenciesStd, self).__init__(name=name)
        self.classes: np.ndarray = np.arange(classes) if isinstance(classes, int) else classes

    def __call__(self, x: Matrix, y: Vector, p: Vector) -> float:
        # bincount is similar to np.unique(..., return_counts=True) but allows to fix a minimum number of classes
        # in this way, if the predictions are all the same, the counts will be [n, 0, ..., 0] instead of [n]
        minlength = 1 + max(np.max(self.classes), np.max(y), np.max(p))
        classes_counts = np.bincount(p, minlength=minlength)
        classes_counts = classes_counts if self.classes is None else classes_counts[self.classes]
        classes_counts = classes_counts / len(p)
        return classes_counts.std()


class MonotonicViolation(Metric):
    """Violation of the Monotonicity Shape Constraint.

    Args:
        monotonicities: the list of pair of indices for which a monotonicity constraint must hold.
        aggregation: the aggregation type:
                       - 'average', which computes the average constraint violation in terms of pure output
                       - 'percentage', which computes the constraint violation in terms of average number of violations
                       - 'feasible', which returns a binary value depending on whether there is at least on violation
        eps: the slack value under which a violation is considered to be acceptable
        name: the name of the metric.
    """

    def __init__(self, monotonicities: Monotonicities, aggregation: str = 'average', eps: float = 1e-3,
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

    def __call__(self, x: Matrix, y: Vector, p: Vector) -> float:
        violations = np.array([0]) if len(self.higher_indices) == 0 else p[self.lower_indices] - p[self.higher_indices]
        violations[violations < self.eps] = 0.0
        return self.aggregate(violations)
