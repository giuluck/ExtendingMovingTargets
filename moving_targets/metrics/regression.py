"""Regression Metrics."""

from typing import Callable

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from moving_targets.metrics.metric import Metric
from moving_targets.util.typing import Matrix, Vector


class RegressionMetric(Metric):
    """Basic interface for a Moving Target's regression metric.

        Args:
            metric_function: callable function that computes the metric given the true targets and the predictions.
            name: the name of the metric.
        """

    def __init__(self, metric_function: Callable, name: str):
        super(RegressionMetric, self).__init__(name=name)
        self.metric_function = metric_function

    def __call__(self, x: Matrix, y: Vector, p: Vector) -> float:
        return self.metric_function(y, p)


class MSE(RegressionMetric):
    """Mean Squared Error Loss.

    Args:
        name: the name of the metric.
    """

    def __init__(self, name: str = 'mse'):
        super(MSE, self).__init__(metric_function=mean_squared_error, name=name)


class MAE(RegressionMetric):
    """Mean Absolute Error Loss.

    Args:
        name: the name of the metric.
    """

    def __init__(self, name: str = 'mae'):
        super(MAE, self).__init__(metric_function=mean_absolute_error, name=name)


class R2(RegressionMetric):
    """R2 Score.

    Args:
        name: the name of the metric.
    """

    def __init__(self, name: str = 'r2'):
        super(R2, self).__init__(metric_function=r2_score, name=name)
