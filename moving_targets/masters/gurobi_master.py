"""Gurobi Master interface."""
import logging
from abc import ABC
from typing import Any, Dict

import numpy as np
from gurobipy import Model, Env, GRB

from moving_targets.masters.losses import LossesHandler
from moving_targets.masters.master import Master
from moving_targets.util.typing import Iteration, Solution

EPS: float = 1e-9
"""Floating point value used in lower/upper bounds to avoid infeasibility due to numeric errors."""


def _abs(model, vector):
    """Gurobi custom `abs_fn` function."""
    abs_vector = []
    for v in vector.flatten():
        aux_v = model.addVar(vtype=GRB.CONTINUOUS, name=f'aux({v})', lb=-float('inf'), column=None, obj=0)
        abs_v = model.addVar(vtype=GRB.CONTINUOUS, name=f'abs({v})', column=None, obj=0)
        model.addConstr(aux_v == v, name=f'aux({v})')
        model.addGenConstrAbs(abs_v, aux_v, name=f'abs({v})')
        abs_vector.append(abs_v)
    return np.array(abs_vector).reshape(vector.shape)


def _log(model, vector):
    """Gurobi custom `log_fn` function."""
    log_vector = []
    for v in vector.flatten():
        aux_v = model.addVar(vtype=GRB.CONTINUOUS, name=f'aux({v})', lb=-float('inf'), column=None, obj=0)
        log_v = model.addVar(vtype=GRB.CONTINUOUS, name=f'log({v})', lb=-float('inf'), column=None, obj=0)
        model.addConstr(aux_v == v, name=f'aux({v})')
        model.addGenConstrExp(log_v, aux_v, name=f'log({v})')
        log_vector.append(log_v)
    return np.array(log_vector).reshape(vector.shape)


class GurobiMaster(Master, ABC):
    """Master interface to Gurobi solver."""

    losses: LossesHandler = LossesHandler(abs_fn=_abs, log_fn=_log)
    """The `LossesHandler` object for this backend solver."""

    def __init__(self, alpha: float, beta: float, verbose: bool, **solver_args):
        """
        :param alpha:
            The non-negative real number which is used to calibrate the two losses in the alpha step.

        :param beta:
            The non-negative real number which is used to constraint the p_loss in the beta step.

        :param verbose:
            Whether or not to print information during the optimization process.

        :param solver_args:
            Parameters of the solver to be set via the `model.SetParam()` function.
        """
        super(GurobiMaster, self).__init__(alpha=alpha, beta=beta)

        self.solver_args: Dict[str, Any] = solver_args
        """Parameters of the solver to be set via the `model.SetParam()` function."""

        self.verbose: bool = verbose
        """Whether or not to print information during the optimization process."""

    def adjust_targets(self, macs, x, y, iteration: Iteration) -> Solution:
        """Leverages the other object methods (build_model, y_loss, p_loss, beta_step)
        in order to build the Gurobi model  and return the adjusted targets.

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
        with Env(empty=True) as env:
            if not self.verbose:
                env.setParam('OutputFlag', 0)
            env.start()
            with Model(env=env, name='model') as model:
                for param, value in self.solver_args.items():
                    model.setParam(param, value)
                model_info = self.build_model(macs, model, x, y, iteration)
                # algorithm core: check for feasibility and behave depending on that
                y_loss = self.y_loss(macs, model, x, y, model_info, iteration)
                p_loss = self.p_loss(macs, model, x, y, model_info, iteration)
                model.update()
                if self.beta_step(macs, model, x, y, model_info, iteration):
                    model.addConstr(p_loss <= self.beta, name='loss')
                    model.setObjective(y_loss, GRB.MINIMIZE)
                else:
                    model.setObjective(y_loss + (1.0 / self.alpha) * p_loss, GRB.MINIMIZE)
                # run the optimization procedure (if the time limit expires, tries to reach at least one solution)
                model.optimize()
                if model.Status == GRB.TIME_LIMIT:
                    model.setParam('TimeLimit', GRB.INFINITY)
                    model.setParam('SolutionLimit', 1)
                    model.optimize()
                # if no solution can be found due to, e.g., infeasibility, no labels are returned
                if model.SolCount == 0:
                    logging.warning(f'Status {model.Status} returned at iteration {iteration}, stop training.')
                    return None
                return self.return_solutions(macs, model, x, y, model_info, iteration)
