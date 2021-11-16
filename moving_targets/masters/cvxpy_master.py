"""Cvxpy Master interface."""
import logging
from abc import ABC
from typing import Any, Dict, Optional

import cvxpy as cp

from moving_targets.masters.losses import LossesHandler
from moving_targets.masters.master import Master
from moving_targets.util.typing import Matrix, Vector, Iteration


# noinspection PyUnusedLocal
def _log(model, x, lb: Optional[float] = None, ub: Optional[float] = None) -> Any:
    """Cvxpy custom `log_fn` function."""
    raise ValueError('Cvxpy Master cannot deal with logarithms.')


class CvxpyMaster(Master, ABC):
    """Master interface to Cvxpy solver."""

    losses = LossesHandler(sum_fn=lambda model, x, lb=None, ub=None: sum(x),
                           abs_fn=lambda model, x, lb=None, ub=None: cp.abs(x),
                           log_fn=lambda model, x, lb=None, ub=None: cp.log(x))
    """The `LossesHandler` object for this backend solver."""

    def __init__(self, alpha: float = 1., beta: float = 1., solver: Optional[str] = 'SCS', **solver_args):
        """
        :param alpha:
            The non-negative real number which is used to calibrate the two losses in the alpha step.

        :param beta:
            The non-negative real number which is used to constraint the p_loss in the beta step.

        :param solver:
            The name of the solver (e.g., SCS, ...).

        :param solver_args:
            Parameters of the solver to be passed to the `model.solve()` function.
        """
        super(CvxpyMaster, self).__init__(alpha=alpha, beta=beta)

        self.solver: Optional[str] = solver
        """The name of the solver (e.g., SCS, ...)."""

        self.solver_args: Dict[str, Any] = solver_args
        """Parameters of the solver to be passed to the `model.solve()` function."""

    def adjust_targets(self, macs, x: Matrix, y: Vector, iteration: Iteration) -> Any:
        """Leverages the other object methods (build_model, y_loss, p_loss, beta_step)
        in order to build the CvxPy model  and return the adjusted targets.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param iteration:
            The current `MACS` iteration, usually a number.

        :returns:
            The output of the `self.return_solutions()` method.
        """
        # build model and get losses
        constraints = []
        model_info = self.build_model(macs, constraints, x, y, iteration)
        # algorithm core: check for feasibility and behave depending on that
        y_loss = self.y_loss(macs, constraints, model_info, x, y, iteration)
        p_loss = self.p_loss(macs, constraints, model_info, x, y, iteration)
        if self.beta_step(macs, constraints, model_info, x, y, iteration):
            constraints.append(p_loss - self._beta <= 0)
            objective = cp.Minimize(y_loss)
        else:
            objective = cp.Minimize(y_loss + (1.0 / self._alpha) * p_loss)
        # solve the problem and get the adjusted labels
        model = cp.Problem(objective, constraints)
        model.solve(solver=self.solver, **self.solver_args)
        if model.status in ['infeasible', 'unbounded']:
            logging.warning(f'Status {model.status} returned at iteration {iteration}, stop training.')
            return None
        return self.return_solutions(macs, model, model_info, x, y, iteration)
