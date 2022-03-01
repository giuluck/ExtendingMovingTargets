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
    - 'degree': the polynomial degree to cancel higher-order effects.
    - 'eps': the tolerance used to cancel higher-order effects.
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

    def build(self, x, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        ---------------------------------------------------------------------------------------------------------------
        
        APPROACH 1:
        
        we build a linear regression for each one of the monotonic features by solving the least-squares method:
              a.T @ a @ w = a.T @ y
        where a = x[c] is the data about a specific feature (column) and w is the respective (scalar) weight which
        is constrained to be either positive or negative depending on the kind of expected monotonicity
        
                ---------------------------------------------------------------------------------------

        v = self.backend.add_variables(*y.shape, vtype='continuous', lb=self.lb, ub=self.ub, name='y')
        for c, d in self.directions.items():
            a, w = x[c].values, self.backend.add_continuous_variable(name=f'w_{c}')
            self.backend.add_constraints([d * w >= 0, (a.T @ a) * w == self.backend.sum(a.T * v)])
        return v
        
                ---------------------------------------------------------------------------------------
        
        the problem is that a linear regressor can only capture the trend (which, moreover, is generally satisfied by 
        default, thus there is no actual constraint to enforce) but we are interested in the satisfaction of the 
        constraint between each pair of variables 
        
        ---------------------------------------------------------------------------------------------------------------
        
        APPROACH 2:

        we build a linear regression (for each monotonic feature) between each couple of data point (x[i], x[j])
        
                ---------------------------------------------------------------------------------------

        variables = self.backend.add_continuous_variables(*y.shape, lb=self.lb, ub=self.ub, name='y')
        for c, d in self.directions.items():
            feature = x[c].values
            weights = self.backend.add_continuous_variables(feature.size * (feature.size - 1) // 2, 2, name=f'w_{c}')
            counter = 0
            constraints = []
            for i, f1, v1 in zip(np.arange(feature.size), feature, variables):
                for f2, v2 in zip(feature[i + 1:], variables[i + 1:]):
                    w = weights[counter]
                    v = np.array([v1, v2])
                    a = np.array([[1, f1], [1, f2]])
                    lr_lhs = np.dot((a.T @ a), w)
                    lr_rhs = self.backend.dot(a.T, v)
                    constraints.append(d * w[1] >= 0)
                    constraints += [lrl == lrr for lrl, lrr in zip(lr_lhs, lr_rhs)]
                    counter += 1
            self.backend.add_constraints(constraints)
        return variables

                ---------------------------------------------------------------------------------------

        the main problem here is scalability, but there is as well a problem in the definition of the constraint since,
        for multivariate inputs, it is not clear whether it makes sense to impose the constraint on each single feature
        because the output is somehow warped in a weird way
        
        ---------------------------------------------------------------------------------------------------------------

        APPROACH 3:

        we build a linear regression for each couple of data points including all the features

                ---------------------------------------------------------------------------------------

        ns, nf = x.shape
        variables = self.backend.add_continuous_variables(ns, lb=self.lb, ub=self.ub, name='y')
        weights = self.backend.add_continuous_variables(ns * (ns - 1) // 2, nf + 1, name=f'w')
        counter = 0
        constraints = []
        for i, x1, v1 in zip(np.arange(ns), x.values, variables):
            for x2, v2 in zip(x.values[i + 1:], variables[i + 1:]):
                w = weights[counter]
                v = np.array([v1, v2])
                a = np.array([[1] + list(x1), [1] + list(x2)])
                lr_lhs = np.dot((a.T @ a), w)
                lr_rhs = self.backend.dot(a.T, v)
                constraints += [lrl == lrr for lrl, lrr in zip(lr_lhs, lr_rhs)]
                counter += 1
        weights = pd.DataFrame(weights, columns=['intercept'] + list(x.columns))
        for c, d in self.directions.items():
            constraints += [w >= 0 for w in d * weights[c]]
        self.backend.add_constraints(constraints)
        return variables.reshape(y.shape)

                ---------------------------------------------------------------------------------------

        it does not work, similarly to the previous approach
        """
        v = self.backend.add_continuous_variables(*y.shape, lb=self.lb, ub=self.ub, name='y')
        for c, d in self.directions.items():
            a = PolynomialFeatures(degree=self.degree).fit_transform(x[[c]])
            w = self.backend.add_continuous_variables(a.shape[1])
            lr_lhs = np.dot((a.T @ a), w)
            lr_rhs = self.backend.dot(a.T, v)
            self.backend.add_constraints([lrl == lrr for lrl, lrr in zip(lr_lhs, lr_rhs)])
            self.backend.add_constraints([w[i] <= self.eps for i in range(2, self.degree + 1)])
            self.backend.add_constraints([w[i] >= -self.eps for i in range(2, self.degree + 1)])
            self.backend.add_constraint(d * w[1] >= 0)
        return v


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
