"""Cplex Master interface."""

from abc import ABC
from docplex.mp.model import Model

from moving_targets.masters.losses import LossesHandler
from moving_targets.masters.master import Master
from moving_targets.utils.typing import Matrix, Vector, Iteration


class CplexMaster(Master, ABC):
    """Master interface to Cplex solver.

    Args:
        time_limit: the maximal time for which the master can run during each iteration.
        **kwargs: super-class arguments.
    """

    losses = LossesHandler(sum_fn=lambda model, x: model.sum(x), abs_fn=lambda model, x: model.abs(x), log_fn=None)

    def __init__(self, time_limit: float = 30., **kwargs):
        super(CplexMaster, self).__init__(**kwargs)
        self.time_limit: float = time_limit

    # noinspection PyMissingOrEmptyDocstring
    def adjust_targets(self, macs, x: Matrix, y: Vector, iteration: Iteration) -> object:
        # build model and get losses
        model = Model(name='model')
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
            raise RuntimeError('The given model has no admissible solution, please check its constraints.')
        return self.return_solutions(macs, solution, model_info, x, y, iteration)
