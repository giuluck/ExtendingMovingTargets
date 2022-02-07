from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd
from moving_targets import MACS
from moving_targets.callbacks import Callback
from moving_targets.learners import MultiLayerPerceptron, Learner
from moving_targets.masters import Master
from moving_targets.masters.backends import CplexBackend
from moving_targets.masters.losses import HammingDistance, CrossEntropy, SAE, SSE, MAE, MSE, aliases
from moving_targets.masters.optimizers import Optimizer
from moving_targets.metrics import Metric
from moving_targets.util.typing import Dataset

from src.datasets import Manager
from src.models import Model
from src.util.metrics import GridConstraint


class MonotonicityMaster(Master):
    """Master for the monotonicity enforcement problem.

    - 'directions' is a dictionary pairing each monotonic attribute to its direction (-1 or 1).
    - 'classification' is True for (binary) classification tasks and False for regression tasks.
    """

    __name__: str = 'MT'

    def __init__(self, directions: Dict[str, int], classification: bool, alpha: float, y_loss: str, p_loss: str):
        y_class, p_class = aliases.get(y_loss), aliases.get(p_loss)
        assert y_class is not None, f"Unknown y_loss '{y_loss}'"
        assert p_class is not None, f"Unknown p_loss '{p_loss}'"
        if classification:
            # check that the loss is a valid classification loss that can use continuous targets
            assert y_class not in [HammingDistance, CrossEntropy], f"Unsupported y_loss '{y_loss}'"
            assert p_class not in [HammingDistance, CrossEntropy], f"Unsupported y_loss '{p_loss}'"
            y_loss, p_loss, lb, ub = y_class(binary=False), p_class(binary=False), 0, 1
        else:
            # check that the loss is a valid regression loss
            assert y_class in [SAE, SSE, MAE, MSE], f"Unsupported y_loss '{y_loss}'"
            assert p_class in [SAE, SSE, MAE, MSE], f"Unsupported p_loss '{p_loss}'"
            y_loss, p_loss, lb, ub = y_class(), p_class(), -float('inf'), float('inf')

        super().__init__(backend=CplexBackend(time_limit=30),
                         y_loss=y_loss,
                         p_loss=p_loss,
                         alpha=Optimizer(base=alpha),
                         beta=None,
                         stats=False)

        self.directions: Dict[str, int] = directions
        self.lb: float = lb
        self.ub: float = ub

    def build(self, x: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        # we build a linear regression for each one of the monotonic features by solving the least-squares method:
        #       a.T @ a @ w = a.T @ y
        # where a = x[c] is the data about a specific feature (column) and w is the respective (scalar) weight which
        # is constrained to be either positive or negative depending on the kind of expected monotonicity
        v = self.backend.add_variables(*y.shape, vtype='continuous', lb=self.lb, ub=self.ub, name='y')
        for c, d in self.directions.items():
            a, w = x[c].values, self.backend.add_continuous_variable(name=f'w_{c}')
            self.backend.add_constraints([d * w >= 0, (a.T @ a) * w == self.backend.sum(a.T * v)])
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

    def __init__(self,
                 dataset: Manager,
                 init_step: str,
                 alpha: float,
                 y_loss: str,
                 p_loss: str,
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
            epochs=1000,
            verbose=False,
            x_scaler=x_scaler,
            y_scaler=y_scaler
        )
        master = MonotonicityMaster(
            directions=dataset.directions,
            classification=dataset.classification,
            alpha=alpha,
            y_loss=y_loss,
            p_loss=p_loss
        )
        self.macs = CustomMACS(
            master=master,
            learner=learner,
            init_step=init_step,
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
