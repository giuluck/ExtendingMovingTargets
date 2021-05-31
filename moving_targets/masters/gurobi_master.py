"""Gurobi Master interface."""
import logging

import numpy as np
from abc import ABC
from typing import Optional
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
        time_limit: the maximal time for which the master can run during each iteration.
        **kwargs: super-class arguments.
    """

    losses = LossesHandler(sum_fn=lambda model, x, lb=None, ub=None: sum(x), abs_fn=_abs, log_fn=_log)

    def __init__(self, time_limit: Optional[float] = None, verbose: bool = False, **kwargs):
        super(GurobiMaster, self).__init__(**kwargs)
        self.time_limit: Optional[float] = time_limit
        self.verbose: bool = verbose

    # noinspection PyMissingOrEmptyDocstring
    def adjust_targets(self, macs, x: Matrix, y: Vector, iteration: Iteration) -> object:
        # build model and get losses
        with Env(empty=True) as env:
            if not self.verbose:
                env.setParam('OutputFlag', 0)
            env.start()
            with Model(env=env, name='model') as model:
                if self.time_limit is not None:
                    model.setParam('TimeLimit', self.time_limit)
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
                # solve the problem and get the adjusted labels
                model.optimize()
                if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
                    logging.warning(f'Status {model.Status} returned at iteration {iteration}, stop training.')
                    return None
                return self.return_solutions(macs, model, model_info, x, y, iteration)
