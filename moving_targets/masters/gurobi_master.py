"""Gurobi Master interface."""
import logging
from abc import ABC
from typing import Any, Dict, Optional

from gurobipy import Model, Env, GRB

from moving_targets.masters.losses import LossesHandler
from moving_targets.masters.master import Master
from moving_targets.util.typing import Iteration, Solution

EPS: float = 1e-9
"""Floating point value used in lower/upper bounds to avoid infeasibility due to numeric errors."""


def _abs(m, mvs, nvs):
    """Gurobi custom `abs_fn` function."""
    abs_vector = []
    for mv, nv in zip(mvs, nvs):
        aux_var = m.addVar(vtype=GRB.CONTINUOUS, name=f'aux({mv})', lb=-float('inf'), column=None, obj=0)
        abs_var = m.addVar(vtype=GRB.CONTINUOUS, name=f'abs({mv})', column=None, obj=0)
        m.addConstr(aux_var == mv - nv, name=f'aux({mv})')
        m.addGenConstrAbs(abs_var, aux_var, name=f'abs({mv})')
        abs_vector.append(abs_var)
    return abs_vector


def _log(m, mvs, nvs):
    """Gurobi custom `log_fn` function."""
    log_vector = []
    for mv, nv in zip(mvs, nvs):
        aux_var = m.addVar(vtype=GRB.CONTINUOUS, name=f'aux({mv})', lb=-float('inf'), column=None, obj=0)
        log_var = m.addVar(vtype=GRB.CONTINUOUS, name=f'log({mv})', lb=-float('inf'), column=None, obj=0)
        m.addConstr(aux_var == mv, name=f'aux({mv})')
        m.addGenConstrExp(log_var, aux_var, name=f'log({mv})')
        log_vector.append(-nv * log_var)
    return log_vector


class GurobiMaster(Master, ABC):
    """Master interface to Gurobi solver."""

    losses: LossesHandler = LossesHandler(abs_fn=_abs, log_fn=_log)
    """The `LossesHandler` object for this backend solver."""

    def __init__(self, alpha: Optional[float], beta: Optional[float], verbose: bool, **solver_args):
        """
        :param alpha:
            The initial positive real number which is used to calibrate the two losses in the alpha step.

        :param beta:
            The initial non-negative real number which is used to constraint the p_loss in the beta step.

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
        # build model
        with Env(empty=True) as env:
            if not self.verbose:
                env.setParam('OutputFlag', 0)
            env.start()
            with Model(env=env, name='model') as model:
                for param, value in self.solver_args.items():
                    model.setParam(param, value)
                # retrieve info and get losses
                info = self.build_model(macs=macs, model=model, x=x, y=y, iteration=iteration)
                y_loss = self.y_loss(macs=macs, model=model, x=x, y=y, model_info=info, iteration=iteration)
                p_loss = self.p_loss(macs=macs, model=model, x=x, y=y, model_info=info, iteration=iteration)
                model.update()
                # check for feasibility and behave depending on that
                beta = self.beta(macs=macs, model=model, x=x, y=y, model_info=info, iteration=iteration)
                if beta is None:
                    alpha = self.alpha(macs=macs, model=model, x=x, y=y, model_info=info, iteration=iteration)
                    model.setObjective(y_loss + (1.0 / alpha) * p_loss, GRB.MINIMIZE)
                else:
                    model.addConstr(p_loss <= beta, name='loss')
                    model.setObjective(y_loss, GRB.MINIMIZE)
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
                return self.return_solutions(macs=macs, solution=model, x=x, y=y, model_info=info, iteration=iteration)
