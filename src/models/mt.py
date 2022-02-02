from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
from moving_targets import MACS
from moving_targets.callbacks import Callback
from moving_targets.learners import MultiLayerPerceptron
from moving_targets.masters import Master
from moving_targets.masters.backends import CplexBackend
from moving_targets.masters.losses import HammingDistance, CrossEntropy, SAE, SSE, MAE, MSE, aliases
from moving_targets.masters.optimizers import Optimizer
from moving_targets.metrics import Metric
from moving_targets.util.scalers import Scaler
from moving_targets.util.typing import Dataset

from src.models import Model


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


class MT(Model):
    """The Moving Targets model.

    - 'directions' is a dictionary pairing each monotonic attribute to its direction (-1, 0, or 1).
    - 'classification' is True for (binary) classification tasks and False for regression tasks.
    """

    def __init__(self,
                 directions: Dict[str, int],
                 classification: bool,
                 init_step: str,
                 alpha: float,
                 y_loss: str,
                 p_loss: str,
                 iterations: int,
                 metrics: List[Metric],
                 callbacks: List[Callback],
                 val_data: Optional[Dataset],
                 verbose: Union[int, bool],
                 scalers: Tuple[Scaler, Scaler]):
        learner = MultiLayerPerceptron(
            loss='binary_crossentropy' if classification else 'mean_squared_error',
            output_activation='sigmoid' if classification else None,
            hidden_units=[128, 128],
            batch_size=32,
            epochs=1000,
            verbose=False,
            x_scaler=scalers[0],
            y_scaler=scalers[1]
        )
        master = MonotonicityMaster(
            directions=directions,
            classification=classification,
            alpha=alpha,
            y_loss=y_loss,
            p_loss=p_loss
        )
        self.macs = MACS(learner=learner, master=master, init_step=init_step, metrics=metrics, stats=True)
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
