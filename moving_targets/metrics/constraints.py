"""Constraints Metrics."""

from typing import Optional, Callable

import numpy as np

from moving_targets.metrics.metric import Metric
from moving_targets.util.typing import Classes, MonotonicitiesList, Matrix, Vector


class ClassFrequenciesStd(Metric):
    """Standard Deviation of the Class Frequencies, usually constrained to be null."""

    def __init__(self, classes: Optional[Classes] = None, name: str = 'class frequencies std'):
        """
        :param classes:
            The number of classes or a vector of class labels.

        :param name:
            The name of the metric.
        """
        super(ClassFrequenciesStd, self).__init__(name=name)

        self.classes: np.ndarray = np.arange(classes) if isinstance(classes, int) else classes
        """The number of classes or a vector of class labels."""

    def __call__(self, x: Matrix, y: Vector, p: Vector) -> float:
        """Core method used to compute the metric value.

        :param x:
            The input matrix.

        :param y:
            The vector of ground truths.

        :param p:
            The vector of predictions.

        :returns:
            The metric value.
        """
        # bincount is similar to np.unique(..., return_counts=True) but allows to fix a minimum number of classes
        # in this way, if the predictions are all the same, the counts will be [n, 0, ..., 0] instead of [n]
        minlength = 1 + max(np.max(self.classes), np.max(y), np.max(p))
        classes_counts = np.bincount(p, minlength=minlength)
        classes_counts = classes_counts if self.classes is None else classes_counts[self.classes]
        classes_counts = classes_counts / len(p)
        return classes_counts.std()


class MonotonicViolation(Metric):
    """Violation of the Monotonicity Shape Constraint."""

    def __init__(self, monotonicities: MonotonicitiesList,
                 aggregation: str = 'average',
                 eps: float = 1e-3,
                 name: str = 'monotonic violation'):
        """
        :param monotonicities:
            The list of pair of indices for which a monotonicity constraint must hold.

        :param aggregation:
            The aggregation type:

            - 'average', which computes the average constraint violation in terms of pure output.
            - 'percentage', which computes the constraint violation in terms of average number of violations.
            - 'feasible', which returns a binary value depending on whether there is at least on violation.

        :param eps:
            The slack value under which a violation is considered to be acceptable.

        :param name:
            The name of the metric.
        """
        super(MonotonicViolation, self).__init__(name=name)

        self.higher_ind: np.ndarray = np.array([hi for hi, _ in monotonicities])
        """The list of indices that are greater to the respective lower_indices."""

        self.lower_ind: np.ndarray = np.array([li for _, li in monotonicities])
        """The list of indices that are lower to the respective higher_indices."""

        self.eps: float = eps
        """The slack value under which a violation is considered to be acceptable."""

        self.aggregate: Callable
        """The aggregation function."""

        if aggregation == 'average':
            self.aggregate = lambda violations: np.mean(violations)
        elif aggregation == 'percentage':
            self.aggregate = lambda violations: np.mean(violations > 0)
        elif aggregation == 'feasible':
            self.aggregate = lambda violations: int(np.all(violations <= 0))
        else:
            raise ValueError(f"'{aggregation}' is not a valid aggregation kind")

    def __call__(self, x: Matrix, y: Vector, p: Vector) -> float:
        """Core method used to compute the metric value.

        :param x:
            The input matrix.

        :param y:
            The vector of ground truths.

        :param p:
            The vector of predictions.

        :returns:
            The metric value.
        """
        violations = np.array([0]) if len(self.higher_ind) == 0 else p[self.lower_ind] - p[self.higher_ind]
        violations[violations < self.eps] = 0.0
        return self.aggregate(violations)
