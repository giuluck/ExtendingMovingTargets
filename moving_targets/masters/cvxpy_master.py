"""Cvxpy Master interface."""
import logging
from abc import ABC
from typing import Any, Dict

import cvxpy as cp
import numpy as np

from moving_targets.masters.losses import LossesHandler
from moving_targets.masters.master import Master
from moving_targets.util.typing import Iteration, Solution


class CvxpyMaster(Master, ABC):
    """Master interface to Abstract Cvxpy solver."""

    scs_losses: LossesHandler = LossesHandler(abs_fn=lambda model, vector: np.array([cp.abs(v) for v in vector]),
                                              log_fn=lambda model, vector: np.array([cp.log(v) for v in vector]))
    """The `LossesHandler` object for the 'SCS' backend solver."""

    def __init__(self, alpha: float, beta: float, solver: str, **solver_args):
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

        self.solver: str = solver
        """The name of the solver."""

        self.solver_args: Dict[str, Any] = solver_args
        """Parameters of the solver to be passed to the `model.solve()` function."""

    def adjust_targets(self, macs, x, y, iteration: Iteration) -> Solution:
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

        :return:
            The output of the `self.return_solutions()` method.
        """
        # build model and get losses
        constraints = []
        model_info = self.build_model(macs, constraints, x, y, iteration)
        # algorithm core: check for feasibility and behave depending on that
        y_loss = self.y_loss(macs, constraints, x, y, model_info, iteration)
        p_loss = self.p_loss(macs, constraints, x, y, model_info, iteration)
        if self.beta_step(macs, constraints, x, y, model_info, iteration):
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
        return self.return_solutions(macs, model, x, y, model_info, iteration)
