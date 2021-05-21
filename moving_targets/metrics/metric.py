"""Basic Metric Interface."""
from moving_targets.util.typing import Matrix, Vector


class Metric:
    """Basic interface for a Moving Target's metric.

    Args:
        name: the name of the metric.
    """

    def __init__(self, name: str):
        super(Metric, self).__init__()
        self.__name__ = name

    def __call__(self, x: Matrix, y: Vector, p: Vector) -> float:
        pass
