"""Basic Metric Interface."""
from moving_targets.util.typing import Matrix, Vector


class Metric:
    """Basic interface for a Moving Targets metric."""

    def __init__(self, name: str):
        """
        :param name:
            The name of the metric.
        """
        super(Metric, self).__init__()

        self.__name__: str = name
        """The name of the metric."""

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
        pass
