from typing import Optional, Tuple, List, Any, Callable

import numpy as np
from gurobipy import Model as GurobiModel, Env, GRB

from moving_targets.masters import GurobiMaster, LossesHandler
from test.moving_targets.abstract_losses import TestLosses


class TestGurobiLosses(TestLosses):
    """Tests the correctness of Gurobi losses."""

    def _losses(self) -> LossesHandler:
        return GurobiMaster.losses

    def _unsupported(self) -> List[str]:
        return []

    def _objective(self, values: np.ndarray, classes: Optional[Tuple[int, str]], loss: Callable) -> Tuple[Any, List]:
        num_samples = len(values)
        with Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with GurobiModel(env=env) as model:
                if classes is None:
                    variables = model.addVars(num_samples, vtype=GRB.CONTINUOUS, lb=-float('inf'), name='y').values()
                    variables = np.array(variables)
                else:
                    num_classes, kind = classes
                    if num_classes == 2 and kind in ['reversed', 'symmetric']:
                        variables = model.addVars(num_samples, vtype=GRB.CONTINUOUS, name='y').values()
                        variables = np.array(variables)
                    elif num_classes == 2:
                        variables = model.addVars(num_samples, vtype=GRB.BINARY, name='y').values()
                        variables = np.array(variables)
                    elif kind in ['reversed', 'symmetric']:
                        variables = model.addVars(num_samples, num_classes, vtype=GRB.CONTINUOUS, name='y').values()
                        variables = np.array(variables).reshape((num_samples, num_classes))
                    else:
                        variables = model.addVars(num_samples, num_classes, vtype=GRB.BINARY, name='y').values()
                        variables = np.array(variables).reshape((num_samples, num_classes))
                model.update()
                model.addConstrs((var == val for var, val in zip(variables.flatten(), values.flatten())), name='c')
                model.setObjective(loss(model, variables), GRB.MINIMIZE)
                model.optimize()
                return model.objVal
