from moving_targets.metrics import MonotonicViolation


def metrics_summary(model, metric, **kwargs):
    summary = []
    for title, (x, y) in kwargs.items():
        summary.append(f'{metric(y, model.predict(x)):.4} ({title} r2)')
    return ', '.join(summary)


def violations_summary(model, grid, monotonicities):
    p = model.predict(grid)
    avg_violation = MonotonicViolation(monotonicities=monotonicities, aggregation='average', eps=0.0)
    pct_violation = MonotonicViolation(monotonicities=monotonicities, aggregation='percentage', eps=0.0)
    return f'{avg_violation(None, None, p):.4} (avg. violation), {pct_violation(None, None, p):.4} (pct. violation)'
