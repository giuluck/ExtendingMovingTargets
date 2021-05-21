from typing import Callable, Tuple, List

from moving_targets.metrics import MonotonicViolation


def metrics_summary(model, metric: Callable, metric_name: str = None, post_process: Callable = None, **kwargs) -> str:
    summary = []
    metric_name = metric.__name__ if metric_name is None else metric_name
    for title, (x, y) in kwargs.items():
        p = model.predict(x) if post_process is None else post_process(model.predict(x))
        summary.append(f'{metric(y, p):.4} ({title} {metric_name})')
    return ', '.join(summary)


def violations_summary(model, grid, monotonicities: List[Tuple[int, int]]) -> str:
    p = model.predict(grid)
    avg_violation = MonotonicViolation(monotonicities=monotonicities, aggregation='average', eps=0.0)
    pct_violation = MonotonicViolation(monotonicities=monotonicities, aggregation='percentage', eps=0.0)
    return f'{avg_violation(None, None, p):.4} (avg. violation), {pct_violation(None, None, p):.4} (pct. violation)'
