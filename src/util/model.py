"""Model utils."""

from typing import Callable, Dict, Union

from moving_targets.metrics import MonotonicViolation
from moving_targets.util.typing import Data, Matrix, MonotonicitiesList


def metrics_summary(model, metric, metric_name: str = None, post_process: Callable = None, return_type: str = 'str',
                    **kwargs: Data) -> Union[str, Dict[str, float]]:
    """Computes the metrics over a custom set of validation data, then builds a summary.

    Args:
        model: a model object having the 'predict(x)' method.
        metric: a function which can compute a metric over a pair of two vectors, the true and the predicted one.
        metric_name: a custom metric name. If None, the original metric name is used instead.
        post_process: a post-processing function for the predictions, if needed.
        return_type: either 'str' to return the string, or 'dict' to return the dictionary.
        **kwargs: a dictionary of named `Data` arguments.

    Returns:
        Either a dictionary for the metric values or a string representing the evaluation summary.
    """
    summary = {}
    metric_name = metric.__name__ if metric_name is None else metric_name
    for title, (x, y) in kwargs.items():
        p = model.predict(x) if post_process is None else post_process(model.predict(x))
        summary[title] = metric(y, p)
    if return_type in ['dict', 'dictionary']:
        return summary
    elif return_type in ['str', 'string']:
        return ', '.join([f'{value:.4} ({title} {metric_name})' for title, value in summary.items()])
    else:
        ValueError(f"Unsupported return type '{return_type}'")


def violations_summary(model,
                       inputs: Matrix,
                       monotonicities: MonotonicitiesList,
                       return_type: str = 'str') -> Union[str, Dict[str, float]]:
    """Computes the violations over a custom set of validation data, then builds a summary.

    Args:
        model: a model object having the 'predict(x)' method.
        inputs: the matrix/dataframe representing the input space.
        monotonicities: the list of monotonicities.
        return_type: either 'str' to return the string, or 'dict' to return the dictionary.

    Returns:
        Either a dictionary for the metric values or a string representing the evaluation summary.
    """
    p = model.predict(inputs)
    summary = {
        'avg_violation': MonotonicViolation(monotonicities=monotonicities, aggregation='average', eps=0.0),
        'pct_violation': MonotonicViolation(monotonicities=monotonicities, aggregation='percentage', eps=0.0)
    }
    summary = {title: metric(x=[], y=[], p=p) for title, metric in summary.items()}
    if return_type in ['dict', 'dictionary']:
        return summary
    elif return_type in ['str', 'string']:
        return ', '.join([f'{value:.4} ({title})' for title, value in summary.items()])
    else:
        ValueError(f"Unsupported return type '{return_type}'")
