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
    """Cplex custom `log_fn` function."""
    raise ValueError('Cplex Master cannot deal with logarithms.')


class CplexMaster(Master, ABC):
    """Master interface to Cplex solver."""

    losses = LossesHandler(sum_fn=lambda model, x, lb=None, ub=None: model.sum(x),
                           abs_fn=lambda model, x, lb=None, ub=None: model.abs(x),
                           log_fn=_log)
    """The `LossesHandler` object for this backend solver."""

    def __init__(self, alpha: float = 1., beta: float = 1., time_limit: Optional[float] = None):
        """
        :param alpha:
            The non-negative real number which is used to calibrate the two losses in the alpha step.

        :param beta:
            The non-negative real number which is used to constraint the p_loss in the beta step.

        :param time_limit:
            The maximal time for which the master can run during each iteration.
        """
        super(CplexMaster, self).__init__(alpha=alpha, beta=beta)

        self.time_limit: Optional[float] = time_limit
        """The maximal time for which the master can run during each iteration."""

    def adjust_targets(self, macs, x: Matrix, y: Vector, iteration: Iteration) -> Any:
        """Leverages the other object methods (build_model, y_loss, p_loss, beta_step) in order to build the Cplex
        model and return the adjusted targets.

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
        model = Model(name='model')
        if self.time_limit is not None:
            model.set_time_limit(self.time_limit)
        model_info = self.build_model(macs=macs, model=model, x=x, y=y, iteration=iteration)
        # algorithm core: check for feasibility and behave depending on that
        y_loss = self.y_loss(macs=macs, model=model, model_info=model_info, x=x, y=y, iteration=iteration)
        p_loss = self.p_loss(macs=macs, model=model, model_info=model_info, x=x, y=y, iteration=iteration)
        if self.beta_step(macs=macs, model=model, model_info=model_info, x=x, y=y, iteration=iteration):
            model.add(p_loss <= self._beta)
            model.minimize(y_loss)
        else:
            model.minimize(y_loss + (1.0 / self._alpha) * p_loss)
        # solve the problem and get the adjusted labels
        solution = model.solve()
        if solution is None:
            logging.warning(f'Model is infeasible at iteration {iteration}, stop training.')
            return None
        return self.return_solutions(macs=macs, solution=solution, model_info=model_info, x=x, y=y, iteration=iteration)
