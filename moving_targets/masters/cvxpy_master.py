"""Cvxpy Master interface."""
import logging
from abc import ABC
from typing import Any, Dict, Optional as Opt

import cvxpy as cp

from moving_targets.masters.losses import LossesHandler
from moving_targets.masters.master import Master
from moving_targets.util.typing import Matrix, Vector, Iteration


# noinspection PyUnusedLocal
def _log(model, x, lb: Opt[float] = None, ub: Opt[float] = None) -> Any:
    raise ValueError('Cvxpy Master cannot deal with logarithms.')


class CvxpyMaster(Master, ABC):
    """Master interface to Cvxpy solver.

    Args:
        solver: the type of solver.
        **solver_args: parameters of the solver to be passed to the `model.solve()` function.
    """

    losses = LossesHandler(sum_fn=lambda model, x, lb=None, ub=None: sum(x),
                           abs_fn=lambda model, x, lb=None, ub=None: cp.abs(x),
                           log_fn=lambda model, x, lb=None, ub=None: cp.log(x))

    def __init__(self, alpha: float = 1., beta: float = 1., solver: Opt[str] = 'SCS', **solver_args):
        super(CvxpyMaster, self).__init__(alpha=alpha, beta=beta)
        self.solver: Opt[str] = solver
        self.solver_args: Dict[str, Any] = solver_args

    # noinspection PyMissingOrEmptyDocstring
    def adjust_targets(self, macs, x: Matrix, y: Vector, iteration: Iteration) -> Any:
        # build model and get losses
        constraints = []
        model_info = self.build_model(macs, constraints, x, y, iteration)
        # algorithm core: check for feasibility and behave depending on that
        y_loss = self.y_loss(macs, constraints, model_info, x, y, iteration)
        p_loss = self.p_loss(macs, constraints, model_info, x, y, iteration)
        if self.beta_step(macs, constraints, model_info, x, y, iteration):
            constraints.append(p_loss - self.beta <= 0)
            objective = cp.Minimize(y_loss)
        else:
            objective = cp.Minimize(y_loss + (1.0 / self.alpha) * p_loss)
        # solve the problem and get the adjusted labels
        model = cp.Problem(objective, constraints)
        model.solve(solver=self.solver, **self.solver_args)
        if model.status in ['infeasible', 'unbounded']:
            logging.warning(f'Status {model.status} returned at iteration {iteration}, stop training.')
            return None
        return self.return_solutions(macs, model, model_info, x, y, iteration)
