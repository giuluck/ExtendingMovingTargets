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
    """

    def __init__(self, classification: bool, directions: Dict[str, int], degree: int, loss: str):
        assert degree > 0, f"'degree' should be a positive integer, got {degree}"

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
        x = x['price'].values
        v = self.backend.add_continuous_variables(*y.shape, lb=self.lb, ub=self.ub, name='y')
        self.backend.model.update()
        constraints = []
        v_mean = self.backend.mean(v)
        for i in np.arange(1, self.degree + 1):
            x_i = x ** i
            cov_v = self.backend.mean(x_i * v) - x_i.mean() * v_mean
            cov_x = np.mean(x_i * x) - x_i.mean() * x.mean()
            constraints.append(cov_v / cov_x)
        self.backend.add_constraints([constraints[0] <= 0] + [constraints[0] == ci for ci in constraints[1:]])

        alpha = self.alpha(x, y, p)
        nabla_term, squared_term = self.loss(self.backend, v, y, p, sample_weight)
        self.backend.minimize(alpha * nabla_term + squared_term)
        z = self.backend.get_values(v) if self.backend.solve().solution is not None else None

        a = PolynomialFeatures(degree=self.degree).fit_transform(x.reshape((-1, 1)))
        c_mt = [self.backend.get_value(c) for c in constraints]
        c_ref = [np.cov(x ** i, z)[0, 1] / np.cov(x ** i, x)[0, 1] for i in range(1, self.degree + 1)]
        # print('Iteration       :', self._macs.iteration)
        # print('MT  constraints :', c_mt)
        # print('Ref constraints :', c_ref)
        # print('Cst difference  :', [m - r for m, r in zip(c_mt, c_ref)])
        # print()
        self.log(c_mt=c_mt, c_ref=c_ref)

        w_lr, _, _, _ = np.linalg.lstsq(a, z, rcond=None)
        w_mt = np.array([z.mean() - c_mt[0] * x.mean(), c_mt[0]] + [0] * (self.degree - 1))
        # from sklearn.metrics import r2_score
        # print('Iteration    :', self._macs.iteration)
        # print('MT weights   :', w_mt)
        # print('MT loss      :', r2_score((a.T @ a) @ w_mt, a.T @ z))
        # print('LR weights   :', w_lr)
        # print('LR loss      :', r2_score((a.T @ a) @ w_lr, a.T @ z))
        # print('Weights diff :', w_mt - w_lr)
        # print()
        self.log(w_lr=w_lr, w_mt=w_mt)

        self.backend.clear()
        return z


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
                 iterations: int,
                 callbacks: List[Callback],
                 val_data: Optional[Dataset],
                 verbose: Union[int, bool]):
        x_scaler, y_scaler = dataset.scalers()
        learner = MultiLayerPerceptron(
            loss='binary_crossentropy' if dataset.classification else 'mean_squared_error',
            output_activation='sigmoid' if dataset.classification else None,
            hidden_units=[8, 8],
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
        # from moving_targets.util.scalers import Scaler
        # xsc, ysc = Scaler('norm'), Scaler('norm')
        # x, y = xsc.fit_transform(x), ysc.fit_transform(y)
        # self.val_data = {k: (xsc.transform(x), ysc.transform(y)) for k, (x, y) in self.val_data.items()}
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
