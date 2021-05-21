"""Model utils."""

from typing import Callable

from moving_targets.metrics import MonotonicViolation
from moving_targets.util.typing import Data, Matrix, Monotonicities


def metrics_summary(model, metric, metric_name: str = None, post_process: Callable = None, **kwargs: Data) -> str:
    """Computes the metrics over a custom set of validation data, then builds a summary.

    Args:
        model: a model object having the 'predict(x)' method.
        metric: a function which can compute a metric over a pair of two vectors, the true and the predicted one.
        metric_name: a custom metric name. If None, the original metric name is used instead.
        post_process: a post-processing function for the predictions, if needed.
        **kwargs: a dictionary of named `Data` arguments.

    Returns:
        A string representing the evaluation summary.
    """
    summary = []
    metric_name = metric.__name__ if metric_name is None else metric_name
    for title, (x, y) in kwargs.items():
        p = model.predict(x) if post_process is None else post_process(model.predict(x))
        summary.append(f'{metric(y, p):.4} ({title} {metric_name})')
    return ', '.join(summary)


def violations_summary(model, grid: Matrix, monotonicities: Monotonicities) -> str:
    """Computes the violations over a custom set of validation data, then builds a summary.

    Args:
        model: a model object having the 'predict(x)' method.
        grid: the matrix/dataframe representing the input space.
        monotonicities: the list of monotonicities.

    Returns:
        A string representing the evaluation summary.
    """
    p = model.predict(grid)
    avg_violation = MonotonicViolation(monotonicities=monotonicities, aggregation='average', eps=0.0)
    pct_violation = MonotonicViolation(monotonicities=monotonicities, aggregation='percentage', eps=0.0)
    # noinspection PyTypeChecker
    return f'{avg_violation(None, None, p):.4} (avg. violation), {pct_violation(None, None, p):.4} (pct. violation)'
