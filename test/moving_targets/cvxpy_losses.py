from typing import Tuple, List, Optional, Callable

import cvxpy as cp
import numpy as np

from moving_targets.masters import CvxpyMaster, LossesHandler
from test.moving_targets.abstract_losses import TestLosses


class TestScsLosses(TestLosses):
    """Tests the correctness of SCS losses."""

    def _losses(self) -> LossesHandler:
        return CvxpyMaster.scs_losses

    def _unsupported(self) -> List[str]:
        # SCS does not support mixed-integer programming
        return ['bh', 'ch', 'bce', 'cce', 'symmetric_bce', 'symmetric_cce']

    def _objective(self, values: np.ndarray, classes: Optional[Tuple[int, str]], loss: Callable) -> float:
        variables, constraints = [], []
        for val in values.flatten():
            var = cp.Variable((1,))
            variables.append(var)
            constraints.append(var == val)
        variables = np.array(variables)
        variables = variables.reshape((len(values), -1)) if classes is not None and classes[0] > 2 else variables
        objective_fn = cp.Minimize(loss(constraints, variables))
        return cp.Problem(objective_fn, constraints).solve()
