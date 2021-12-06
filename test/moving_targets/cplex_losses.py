from typing import Optional, Tuple, List, Callable

import numpy as np
from docplex.mp.model import Model as CplexModel

from moving_targets.masters import CplexMaster, LossesHandler
from test.moving_targets.abstract_losses import TestLosses


class TestCplexLosses(TestLosses):
    """Tests the correctness of Cplex losses."""

    def _losses(self) -> LossesHandler:
        return CplexMaster.losses

    def _unsupported(self) -> List[str]:
        # cplex does not support logarithms
        return ['reversed_bce', 'reversed_cce', 'symmetric_bce', 'symmetric_cce']

    def _objective(self, values: np.ndarray, classes: Optional[Tuple[int, str]], loss: Callable) -> float:
        model = CplexModel()
        num_samples = len(values)
        if classes is None:
            variables = model.continuous_var_list(keys=num_samples, lb=-float('inf'), name='y')
            variables = np.array(variables)
        elif classes[0] == 2:
            variables = model.binary_var_list(keys=num_samples, name='y')
            variables = np.array(variables)
        else:
            variables = model.binary_var_matrix(keys1=num_samples, keys2=classes[0], name='y')
            variables = np.array(list(variables.values())).reshape((num_samples, classes[0]))
        model.add_constraints([var == val for var, val in zip(variables.flatten(), values.flatten())])
        model.minimize(loss(model, variables))
        return model.solve().objective_value
