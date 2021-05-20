from abc import ABC
from typing import Any

from docplex.mp.model import Model

from moving_targets.masters.losses import LossesHandler
from moving_targets.masters.master import Master


class CplexMaster(Master, ABC):
    losses = LossesHandler(sum_fn=lambda model, x: model.sum(x), abs_fn=lambda model, x: model.abs(x), log_fn=None)

    def __init__(self, alpha: float = 1., beta: float = 1., time_limit: float = 30.):
        super(CplexMaster, self).__init__(alpha=alpha, beta=beta)
        self.time_limit: float = time_limit

    def adjust_targets(self, macs, x, y, iteration: int) -> Any:
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
