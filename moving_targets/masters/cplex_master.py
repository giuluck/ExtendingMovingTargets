"""Cplex Master interface."""
import logging
from abc import ABC
from typing import Optional, Any

from docplex.mp.dvar import Var
from docplex.mp.model import Model

from moving_targets.masters.losses import LossesHandler
from moving_targets.masters.master import Master
from moving_targets.util.typing import Matrix, Vector, Iteration


# noinspection PyUnusedLocal
def _log(model, x: Var, lb: Optional[float] = None, ub: Optional[float] = None) -> Var:
    raise ValueError('Cplex Master cannot deal with logarithms.')


class CplexMaster(Master, ABC):
    """Master interface to Cplex solver.

    Args:
        time_limit: the maximal time for which the master can run during each iteration.
        **kwargs: super-class arguments.
    """

    losses = LossesHandler(sum_fn=lambda model, x, lb=None, ub=None: model.sum(x),
                           abs_fn=lambda model, x, lb=None, ub=None: model.abs(x),
                           log_fn=_log)

    def __init__(self, time_limit: Optional[float] = None, **kwargs):
        super(CplexMaster, self).__init__(**kwargs)
        self.time_limit: Optional[float] = time_limit

    # noinspection PyMissingOrEmptyDocstring
    def adjust_targets(self, macs, x: Matrix, y: Vector, iteration: Iteration) -> Any:
        # build model and get losses
        model = Model(name='model')
        if self.time_limit is not None:
            model.set_time_limit(self.time_limit)
        model_info = self.build_model(macs, model, x, y, iteration)
        # algorithm core: check for feasibility and behave depending on that
        y_loss = self.y_loss(macs, model, model_info, x, y, iteration)
        p_loss = self.p_loss(macs, model, model_info, x, y, iteration)
        if self.beta_step(macs, model, model_info, x, y, iteration):
            model.add(p_loss <= self.beta)
            model.minimize(y_loss)
        else:
            model.minimize(y_loss + (1.0 / self.alpha) * p_loss)
        # solve the problem and get the adjusted labels
        solution = model.solve()
        if solution is None:
            logging.warning(f'Model is infeasible at iteration {iteration}, stop training.')
            return None
        return self.return_solutions(macs, solution, model_info, x, y, iteration)
