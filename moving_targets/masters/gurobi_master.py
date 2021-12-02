"""Gurobi Master interface."""
import logging
from abc import ABC
from typing import Optional, Any, Dict

import numpy as np
from gurobipy import Model, Env, GRB, Var

from moving_targets.masters.losses import LossesHandler
from moving_targets.masters.master import Master
from moving_targets.util.typing import Iteration, Solution

EPS: float = 1e-9
"""Floating point value used in lower/upper bounds to avoid infeasibility due to numeric errors."""


def _abs(model: Model, x: Var, lb: Optional[float] = None, ub: Optional[float] = None) -> Var:
    """Gurobi custom `abs_fn` function."""
    lb = -float('inf') if lb is None else lb
    ub = float('inf') if ub is None else ub
    abs_ub = max(abs(lb), abs(ub))
    aux_x = model.addVar(lb=lb - EPS, ub=ub + EPS, vtype=GRB.CONTINUOUS, name=f'aux({x})', column=None, obj=0)
    abs_x = model.addVar(lb=0, ub=abs_ub + EPS, vtype=GRB.CONTINUOUS, name=f'abs({x})', column=None, obj=0)
    model.addConstr(aux_x == x, name=f'aux({x})')
    model.addGenConstrAbs(abs_x, aux_x, name=f'abs({x})')
    return abs_x


def _log(model: Model, x: Var, lb: Optional[float] = None, ub: Optional[float] = None) -> Var:
    """Gurobi custom `log_fn` function."""
    lb = 0 if lb is None else lb
    ub = float('inf') if ub is None else ub
    log_lb = -float('inf') if lb == 0 else np.log(lb)
    log_ub = -float('inf') if ub == 0 else np.log(ub)
    aux_x = model.addVar(lb=lb - EPS, ub=ub + EPS, vtype=GRB.CONTINUOUS, name=f'aux({x})', column=None, obj=0)
    log_x = model.addVar(lb=log_lb - EPS, ub=log_ub + EPS, vtype=GRB.CONTINUOUS, name=f'log({x})', column=None, obj=0)
    model.addConstr(aux_x == x, name=f'aux({x})')
    model.addGenConstrExp(log_x, aux_x, name=f'log({x})', options='')
    return log_x


class GurobiMaster(Master, ABC):
    """Master interface to Gurobi solver."""

    losses = LossesHandler(sum_fn=lambda model, x, lb=None, ub=None: sum(x), abs_fn=_abs, log_fn=_log)
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
