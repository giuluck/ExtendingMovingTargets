from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd
from moving_targets import MACS
from moving_targets.callbacks import Callback
from moving_targets.learners import Learner, MultiLayerPerceptron
from moving_targets.masters import Master
from moving_targets.masters.backends import GurobiBackend
from moving_targets.masters.losses import HammingDistance, MAE, MSE, aliases
from moving_targets.metrics import Metric
from moving_targets.util.typing import Dataset
from sklearn.preprocessing import PolynomialFeatures

from src.datasets import Manager
from src.models import Model
from src.util.metrics import GridConstraint


class MonotonicityMaster(Master):
    """Master for the monotonicity enforcement problem.

    - 'directions' is a dictionary pairing each monotonic attribute to its direction (-1 or 1).
    - 'classification' is True for (binary) classification tasks and False for regression tasks.
    - 'degree' is the polynomial degree to cancel higher-order effects.
    - 'eps' is the tolerance used to cancel higher-order effects.
    """

    def __init__(self, classification: bool, directions: Dict[str, int], degree: int, eps: float, loss: str):
        assert degree > 0, f"'degree' should be a positive integer, got {degree}"
        assert eps > 0, f"'eps' should be a positive real number, got {eps}"

        # 1. check the correctness of the loss name
        # 2. check that either we are in a regression task or the classification loss can use continuous targets
        # 3. check that either we are in a classification task or the loss is a valid regression loss
        loss_cls = aliases.get(loss)
        assert loss_cls is not None, f"Unknown loss '{loss}'"
        assert not classification or loss_cls != HammingDistance, f"Unsupported loss '{loss}' for classification tasks"
        assert classification or loss_cls in [MAE, MSE], f"Unsupported loss '{loss}' for regression tasks"
        loss = loss_cls(binary=False, name=loss)
        super().__init__(backend=GurobiBackend(time_limit=30), loss=loss, alpha='harmonic', stats=False)

        self.directions: Dict[str, int] = {c: d for c, d in directions.items() if d != 0}
        self.lb: float = 0 if classification else -float('inf')
        self.ub: float = 1 if classification else float('inf')
        self.degree: int = degree
        self.eps: float = eps

    # def build(self, x, y, p):
    #     v = self.backend.add_continuous_variables(*y.shape, lb=self.lb, ub=self.ub, name='y')
    #     for c, d in self.directions.items():
    #         a = PolynomialFeatures(degree=self.degree).fit_transform(x[[c]])
    #         w = self.backend.add_continuous_variables(a.shape[1])
    #         lr_lhs = np.dot((a.T @ a), w)
    #         lr_rhs = self.backend.dot(a.T, v)
    #         self.backend.add_constraints([lrl == lrr for lrl, lrr in zip(lr_lhs, lr_rhs)])
    #         self.backend.add_constraints([w[i] <= self.eps for i in range(2, self.degree + 1)])
    #         self.backend.add_constraints([w[i] >= -self.eps for i in range(2, self.degree + 1)])
    #         self.backend.add_constraint(d * w[1] >= 0)
    #     return v

    def build(self, x, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        pass

    def adjust_targets(self,
                       x,
                       y: np.ndarray,
                       p: Optional[np.ndarray],
                       sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        self.backend.build()

        assert len(self.directions) == 1
        c = list(self.directions.keys())[0]
        d = self.directions[c]
        v = self.backend.add_continuous_variables(*y.shape, lb=self.lb, ub=self.ub, name='y')
        a = PolynomialFeatures(degree=self.degree).fit_transform(x[[c]])
        w = self.backend.add_continuous_variables(a.shape[1])
        lr_lhs = np.dot((a.T @ a), w)
        lr_rhs = self.backend.dot(a.T, v)
        self.backend.add_constraints([lrl == lrr for lrl, lrr in zip(lr_lhs, lr_rhs)])
        self.backend.add_constraints([w[i] <= self.eps for i in range(2, self.degree + 1)])
        self.backend.add_constraints([w[i] >= -self.eps for i in range(2, self.degree + 1)])
        self.backend.add_constraint(d * w[1] >= 0)

        alpha = self.alpha(x, y, p)
        nabla_term, squared_term = self.loss(self.backend, v, y, p, sample_weight)
        self.backend.minimize(alpha * nabla_term + squared_term)
        adjusted = self.backend.get_values(v) if self.backend.solve().solution is not None else None

        a = PolynomialFeatures(degree=self.degree).fit_transform(x[['price']])
        w_lr, _, _, _ = np.linalg.lstsq(a, adjusted, rcond=None)
        w_mt = self.backend.get_values(w)
        print('Iteration    :', self._macs.iteration)
        print('MT weights   :', w_mt)
        print('MT loss      :', np.abs((a.T @ a) @ w_mt - a.T @ adjusted).sum())
        print('LR weights   :', w_lr)
        print('LR loss      :', np.abs((a.T @ a) @ w_lr - a.T @ adjusted).sum())
        print('Weights diff :', w_mt - w_lr)
        print()
        self.log(w_lr=w_lr, w_mt=w_mt)

        self.backend.clear()
        return adjusted


class CustomMACS(MACS):
    """Custom MACS implementation to deal with violation metric (which is evaluated on the grid)."""

    def __init__(self,
                 init_step: str,
                 master: Master,
                 learner: Learner,
                 metrics: List[Metric],
                 constraint: GridConstraint,
                 stats: Union[bool, List[str]]):
        super(CustomMACS, self).__init__(master, learner, init_step, metrics, stats)
        self.constraint: GridConstraint = constraint

    def _compute_metrics(self,
                         x,
                         y: np.ndarray,
                         p: np.ndarray,
                         metrics: List[Metric],
                         prefix: Optional[str] = None) -> Dict[str, float]:
        results = super(CustomMACS, self)._compute_metrics(x, y, p, metrics, prefix)
        for aggregation, value in self.constraint(model=self).items():
            results[f'monotonicity/{aggregation}'] = value
        return results


class MT(Model):
    """The Moving Targets model, which leverages the dataset instance to choose the correct metrics and scalers."""

    __name__: str = 'MT'

    def __init__(self,
                 dataset: Manager,
                 loss: str,
                 degree: int,
                 eps: float,
                 iterations: int,
                 callbacks: List[Callback],
                 val_data: Optional[Dataset],
                 verbose: Union[int, bool]):
        x_scaler, y_scaler = dataset.scalers()
        learner = MultiLayerPerceptron(
            loss='binary_crossentropy' if dataset.classification else 'mean_squared_error',
            output_activation='sigmoid' if dataset.classification else None,
            hidden_units=[128, 128],
            batch_size=32,
            epochs=200,
            verbose=False,
            x_scaler=x_scaler,
            y_scaler=y_scaler
        )
        # from moving_targets.learners import LinearRegression, LogisticRegression
        # learner = LogisticRegression() if dataset.classification else LinearRegression()
        master = MonotonicityMaster(
            classification=dataset.classification,
            directions=dataset.directions,
            degree=degree,
            eps=eps,
            loss=loss
        )
        self.macs = CustomMACS(
            master=master,
            learner=learner,
            init_step='pretraining',
            metrics=dataset.metrics,
            constraint=dataset.constraint,
            stats=True
        )
        self.iterations = iterations
        self.callbacks = callbacks
        self.val_data = val_data
        self.verbose = verbose

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> Any:
        # # TODO: rollback
        from moving_targets.util.scalers import Scaler
        xsc, ysc = Scaler('norm'), Scaler('norm')
        x, y = xsc.fit_transform(x), ysc.fit_transform(y)
        self.val_data = {k: (xsc.transform(x), ysc.transform(y)) for k, (x, y) in self.val_data.items()}
        return self.macs.fit(
            x=x,
            y=y,
            verbose=self.verbose,
            val_data=self.val_data,
            callbacks=self.callbacks,
            iterations=self.iterations,
            sample_weight=None
        )

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.macs.predict(x)
