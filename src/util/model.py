"""Model utils."""

from typing import Callable, Dict, Union

import numpy as np

from moving_targets.metrics import MonotonicViolation
from moving_targets.util.typing import Data, Matrix, MonotonicitiesList


def metrics_summary(model,
                    metric, metric_name: str = None,
                    post_process: Callable = None,
                    return_type: str = 'str',
                    **data_splits: Data) -> Union[str, Dict[str, float]]:
    """Computes the metrics over a custom set of validation data, then builds a summary.

    :param model:
        A model object having the 'predict(x)' method.

    :param metric:
        A function which can compute a metric over a pair of two vectors, the true and the predicted one.

    :param metric_name:
        A custom metric name. If None, the original metric name is used instead.

    :param post_process:
        Either None (identity function) or an explicit post-processing function f(p) for the predictions, which may be
        used, e.g., to convert the probabilities into output classes for certain metrics.

    :param return_type:
        Either 'str' to return the string, or 'dict' to return the dictionary.

    :param data_splits:
        A dictionary of named `Data` arguments, where the name of the argument represents the data split and the value
        is a tuple (<input_data>, <ground_truths>), e.g., "train=(x, y)".

    :returns:
        Either a dictionary for the metric values or a string representing the evaluation summary.
    """
    summary = {}
    metric_name = metric.__name__ if metric_name is None else metric_name
    for title, (x, y) in data_splits.items():
        p = model.predict(x) if post_process is None else post_process(model.predict(x))
        summary[title] = metric(y, p.astype(np.float64))
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

    :param model:
        A model object having the 'predict(x)' method.

    :param inputs:
        The matrix/dataframe representing the input space.

    :param monotonicities:
        The list of monotonicities, in the form of [(<higher_index>, <lower_index>), ...].

    :param return_type:
        Either 'str' to return the string, or 'dict' to return the dictionary.

    :returns:
        Either a dictionary for the metric values or a string representing the evaluation summary.
    """
    p = model.predict(inputs).astype(np.float64)
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
