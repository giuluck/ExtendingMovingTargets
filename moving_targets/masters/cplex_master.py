"""Cplex Master interface."""
import logging
from abc import ABC
from typing import Optional

from docplex.mp.model import Model

from moving_targets.masters.losses import LossesHandler
from moving_targets.masters.master import Master
from moving_targets.util.typing import Iteration, Solution


class CplexMaster(Master, ABC):
    """Master interface to Cplex solver."""

    losses: LossesHandler = LossesHandler(sum_fn=lambda m, v: m.sum(v.flatten()),
                                          abs_fn=lambda m, mvs, nvs: [m.abs(v) for v in mvs - nvs],
                                          log_fn=None)
    """The `LossesHandler` object for this backend solver."""

    def __init__(self, alpha: float, beta: float, time_limit: Optional[float]):
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

    def adjust_targets(self, macs, x, y, iteration: Iteration) -> Solution:
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

        # TODO: remove print
        import time
        temp = time.time()
        y_loss = self.y_loss(macs=macs, model=model, x=x, y=y, model_info=model_info, iteration=iteration)
        print('y_loss:', time.time() - temp)
        temp = time.time()
        p_loss = self.p_loss(macs=macs, model=model, x=x, y=y, model_info=model_info, iteration=iteration)
        print('p_loss:', time.time() - temp)
        print()

        if self.beta_step(macs=macs, model=model, x=x, y=y, model_info=model_info, iteration=iteration):
            model.add(p_loss <= self.beta)
            model.minimize(y_loss)
        else:
            model.minimize(y_loss + (1.0 / self.alpha) * p_loss)
        # solve the problem and get the adjusted labels
        solution = model.solve()
        if solution is None:
            logging.warning(f'Model is infeasible at iteration {iteration}, stop training.')
            return None
        return self.return_solutions(macs=macs, solution=solution, x=x, y=y, model_info=model_info, iteration=iteration)
