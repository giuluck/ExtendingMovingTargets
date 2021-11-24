"""Regression Metrics."""

from typing import Callable

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from moving_targets.metrics.metric import Metric
from moving_targets.util.typing import Matrix, Vector


class RegressionMetric(Metric):
    """Basic interface for a Moving Targets regression metric."""

    def __init__(self, metric_function: Callable, name: str):
        """
        :param metric_function:
            Callable function that computes the metric given the true targets and the predictions.

        :param name:
            The name of the metric.
        """
        super(RegressionMetric, self).__init__(name=name)

        self.metric_function: Callable = metric_function
        """The callable function used to compute the metric."""

    def __call__(self, x: Matrix, y: Vector, p: Vector) -> float:
        """Core method used to compute the metric value.

        :param x:
            The input matrix (unused).

        :param y:
            The vector of ground truths.

        :param p:
            The vector of predictions.

        :return:
            The metric value.
        """
        return self.metric_function(y, p)


class MSE(RegressionMetric):
    """Mean Squared Error Loss."""

    def __init__(self, name: str = 'mse'):
        """
        :param name:
            The name of the metric.
        """
        super(MSE, self).__init__(metric_function=mean_squared_error, name=name)


class MAE(RegressionMetric):
    """Mean Absolute Error Loss."""

    def __init__(self, name: str = 'mae'):
        """
        :param name:
            The name of the metric.
        """
        super(MAE, self).__init__(metric_function=mean_absolute_error, name=name)


class R2(RegressionMetric):
    """R2 Score."""

    def __init__(self, name: str = 'r2'):
        """
        :param name:
            The name of the metric.
        """
        super(R2, self).__init__(metric_function=r2_score, name=name)
