"""Gurobi Master interface."""
import logging
from abc import ABC
from typing import Optional, Any, Dict

import numpy as np
from gurobipy import Model, Env, GRB, Var

from moving_targets.masters.losses import LossesHandler
from moving_targets.masters.master import Master
from moving_targets.util.typing import Matrix, Vector, Iteration

EPS = 1e-9  # used in lower/upper bounds to avoid infeasibility due to numeric errors


def _abs(model: Model, x: Var, lb: Optional[float] = None, ub: Optional[float] = None) -> Var:
    lb = -float('inf') if lb is None else lb
    ub = float('inf') if ub is None else ub
    abs_ub = max(abs(lb), abs(ub))
    aux_x = model.addVar(lb=lb - EPS, ub=ub + EPS, vtype=GRB.CONTINUOUS, name=f'aux({x})', column=None, obj=0)
    abs_x = model.addVar(lb=0, ub=abs_ub + EPS, vtype=GRB.CONTINUOUS, name=f'abs({x})', column=None, obj=0)
    model.addConstr(aux_x == x, name=f'aux({x})')
    model.addGenConstrAbs(abs_x, aux_x, name=f'abs({x})')
    return abs_x


def _log(model: Model, x: Var, lb: Optional[float] = None, ub: Optional[float] = None) -> Var:
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
    """Master interface to Gurobi solver.

    Args:
        verbose: whether or not to print information during the optimization process.
        **solver_args: parameters of the solver to be set via the `model.SetParam()` function.
    """

    losses = LossesHandler(sum_fn=lambda model, x, lb=None, ub=None: sum(x), abs_fn=_abs, log_fn=_log)

    def __init__(self, alpha: float = 1., beta: float = 1., verbose: bool = False, **solver_args):
        super(GurobiMaster, self).__init__(alpha=alpha, beta=beta)
        self.solver_args: Dict[str, Any] = solver_args
        self.verbose: bool = verbose

    # noinspection PyMissingOrEmptyDocstring
    def adjust_targets(self, macs, x: Matrix, y: Vector, iteration: Iteration) -> Any:
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
                y_loss = self.y_loss(macs, model, model_info, x, y, iteration)
                p_loss = self.p_loss(macs, model, model_info, x, y, iteration)
                model.update()
                if self.beta_step(macs, model, model_info, x, y, iteration):
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
                return self.return_solutions(macs, model, model_info, x, y, iteration)
