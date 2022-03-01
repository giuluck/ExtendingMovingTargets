from typing import Dict, List, Tuple, Union, Callable

import numpy as np
import pandas as pd
from moving_targets.metrics import MonotonicViolation, Metric
from sklearn.preprocessing import PolynomialFeatures


class GridConstraint:
    """Custom constraint metric to evaluate the monotonicity constraint satisfaction on an explicit grid.

    - 'grid' is the grid input data.
    - 'monotonicities' is the list of expected monotonicities.
    """

    def __init__(self, grid: pd.DataFrame, monotonicities: List[Tuple[int, int]]):
        self.grid: pd.DataFrame = grid
        self.monotonicities: List[Tuple[int, int]] = monotonicities

    def __call__(self, model) -> Dict[str, float]:
        x, y, p, results = self.grid, np.zeros(1), model.predict(self.grid), {}
        for aggregation in ['average', 'percentage']:
            metric = MonotonicViolation(monotonicities_fn=lambda _: self.monotonicities, aggregation=aggregation)
            results[aggregation] = metric(x, y, p)
        return results


class MonotonicityConstraint(Metric):
    """Constraint metric that resembles the moving targets constraint.

    - 'degree' is the polynomial degree to cancel higher-order effects.

    - 'aggregation' is the aggregation policy in case of multiple features, which can be either a string representing a
        numpy method (e.g., 'sum', 'mean', 'max', 'min', 'std'), a custom callable function taking the vector 'w' as
        parameter, or None to get in output the weight for each feature without any aggregation.

    - 'binarize' is a boolean representing whether to consider the weights values or just an indicator representing if
        the constraint has been satisfied (0) or not (1).
    """

    def __init__(self,
                 degree: int,
                 directions: Dict[str, int],
                 aggregation: Union[None, str, Callable] = 'sum',
                 binarize: bool = True,
                 name: str = 'constraint'):
        assert degree > 0, f"'degree' should be a positive integer, got {degree}"

        super(MonotonicityConstraint, self).__init__(name=name)
        self.directions: Dict[str, int] = {c: d for c, d in directions.items() if d != 0}
        self.degree: int = degree

        if aggregation is None:
            # if the given aggregation is None, return the weights as a dictionary indexed by feature
            features = list(self.directions.keys())
            aggregation = lambda w: {f: v for f, v in zip(features, w)}
        elif isinstance(aggregation, str):
            # if the given aggregation is a string representing a numpy method, use np.<method_name>()
            assert hasattr(np, aggregation), f"'{aggregation}' is not a supported aggregation policy"
            aggregation = getattr(np, aggregation)

        self.aggregation: Callable = (lambda w: aggregation(np.sign(w))) if binarize else aggregation

    def __call__(self, x, y: np.ndarray, p: np.ndarray) -> Union[float, Dict[str, float]]:
        weights = []
        for c, d in self.directions.items():
            a = PolynomialFeatures(degree=self.degree).fit_transform(x[[c]])
            w, _, _, _ = np.linalg.lstsq(a, p, rcond=None)
            weights.append(max(0, -w[1] * d))
        return self.aggregation(np.array(weights))
