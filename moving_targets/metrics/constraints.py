"""Constraints Metrics."""

from typing import Optional, Callable

import numpy as np
import pandas as pd

from moving_targets.learners.learner import Classifier
from moving_targets.metrics.metric import Metric
from moving_targets.util.typing import Classes


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

    def __call__(self, x, y, p) -> float:
        # handle classes and retrieve labels from probabilities
        c = [] if self.classes is None else self.classes
        p = Classifier.get_classes(p)
        # bincount is similar to np.unique(..., return_counts=True) but allows to fix a minimum number of classes
        # in this way, if the predictions are all the same, the counts will be [n, 0, ..., 0] instead of [n]
        minlength = 1 + np.max(np.concatenate((c, y, p)).astype(int))
        classes_counts = np.bincount(p, minlength=minlength)
        classes_counts = classes_counts if self.classes is None else classes_counts[self.classes]
        classes_counts = classes_counts / len(p)
        return classes_counts.std()


class DIDI(Metric):
    """Disparate Impact Discrimination Index."""

    def __init__(self, classification: bool, protected: str, name: str = 'didi percentage'):
        """
        :param classification:
            Whether the task is a classification (True) or a regression (False) task.

        :param protected:
            The name of the protected feature.

            During the solving process, the algorithm checks inside the data for column names starting with the given
            feature name: if a single column is found this is assumed to be a categorical column (thus, the number of
            classes is inferred from the unique values in the column), otherwise, if multiple columns are found this is
            assumed to be a one-hot encoded version of the categorical column (thus, the number of classes is inferred
            from the number of columns).

        :param name:
            The name of the metric.
        """
        super(DIDI, self).__init__(name=name)

        self.classification: bool = classification
        """Whether the task is a classification (True) or a regression (False) task."""

        self.protected: str = protected
        """The name of the protected feature."""

    def __call__(self, x: pd.DataFrame, y, p) -> float:
        didi = 0.0
        targets = Classifier.get_classes(p) if self.classification else p
        indicator_matrix = self.get_indicator_matrix(x=x)
        for label in indicator_matrix.columns:
            # subset of the targets having <label> as protected feature (i.e., the current protected group)
            protected_targets = targets[indicator_matrix[label] == 1]
            if len(protected_targets) > 0:
                if self.classification:
                    # array of all the output classes
                    classes = np.unique(y)
                    # list of deviations from the total percentage of samples respectively to each target class
                    deviation_per_class = [np.mean(protected_targets == c) - np.mean(targets == c) for c in classes]
                    # total deviation (partial didi) respectively to each protected group
                    didi += np.abs(deviation_per_class).sum()
                else:
                    # total deviation (partial didi) respectively to each protected group
                    didi += abs(np.mean(protected_targets) - np.mean(targets))
        return didi

    def get_indicator_matrix(self, x: pd.DataFrame) -> pd.DataFrame:
        """Computes the indicator matrix given the input data and a protected feature.

        :param x:
            The input data (it must be a pandas dataframe because features are searched by column name).

        :return:
            A dataframe representing the indicator matrix.

        :raise `ValueError`:
            If there is no column name starting with the `protected` string.
        """
        protected = [c for c in x.columns if c.startswith(self.protected)]
        if len(protected) == 1:
            # if a single protected column is found out, this is interpreted as a categorical column
            feature = x[protected[0]]
            classes = np.unique(feature)
            return pd.concat(([pd.Series(feature == i, name=i, dtype=int) for i in classes]), axis=1)
        if len(protected) > 1:
            # if multiple protected columns are found out, these are interpreted as one-hot encoded columns
            return x[protected].astype(int)
        raise ValueError(f"No column in {x.columns} starts with the given protected feature '{protected}'")


class MonotonicViolation(Metric):
    """Violation of the Monotonicity Shape Constraint."""

    def __init__(self,
                 monotonicities_fn: Callable,
                 aggregation: str = 'average',
                 eps: float = 1e-3,
                 name: str = 'monotonic violation'):
        """
        :param monotonicities_fn:
            Function having signature f(x) -> M where x is the input data and M is the monotonicities matrix, i.e., a
            NxN matrix (where |x| = N) which associates to each entry (i, j) a value of -1, 0, o 1 depending on the
            expected monotonicity between x[i] and x[j].

            E.g., if x = [0, 1, 2, 3], with 0 < 1 < 2 < 3, then M =
                |  0, -1, -1, -1 |
                |  1,  0, -1, -1 |
                |  1,  1,  0, -1 |
                |  1,  1,  1,  0 |

            If the computation of the monotonicities is heavy and you call the metric multiple times with the same
            input, it would be better to precompute the matrix on the input x and pass a fake function that ignores
            the x parameter and just returns the precomputed matrix, e.g:

            .. code-block:: python

                M = monotonicities_fn(x)
                metric = MonotonicViolation(monotonicities_fn=lambda x: M)

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

        self.monotonicities_fn: Callable = monotonicities_fn
        """Function having signature f(x) -> M where x is the input data and M is the monotonicities matrix."""

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

    def __call__(self, x, y, p) -> float:
        monotonicities = self.monotonicities_fn(x)
        if np.all(monotonicities == 0):
            return 0.0
        # get pair of higher indices and lower indices by filtering pairs with increasing monotonicities
        monotonicities = [(hi, li) for hi, row in enumerate(monotonicities) for li, m in enumerate(row) if m == 1]
        # compute violations as the difference between p[lower_indices] and p[higher_indices]
        violations = p[[li for _, li in monotonicities]] - p[[hi for hi, _ in monotonicities]]
        # then filter out values under the threshold
        violations[violations < self.eps] = 0.0
        return self.aggregate(violations)
