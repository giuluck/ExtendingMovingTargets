from typing import Optional
from docplex.mp.model import Model as CPModel
from docplex.mp.solution import SolveSolution as CPSol

from src.moving_targets.macs import MACS
from src.moving_targets.masters import Master


class CplexMaster(Master):
    def __init__(self, alpha: float = 1., beta: float = 1., time_limit: float = 30.):
        super(CplexMaster, self).__init__(alpha, beta)
        self.time_limit = time_limit

    def define_variables(self, macs: MACS, model: CPModel, x, y, iteration) -> object:
        raise NotImplementedError("Please implement method 'define_model_variables'")

    def compute_losses(self, macs: MACS, model: CPModel, variables, x, y, iteration) -> object:
        raise NotImplementedError("Please implement method 'compute_losses'")

    def return_solutions(self, macs: MACS, solution: Optional[CPSol], variables, x, y, iteration) -> object:
        raise NotImplementedError("Please implement method 'return solutions'")

    def adjust_targets(self, macs, x, y, iteration):
        # build model and get losses
        model = CPModel()
        model.set_time_limit(self.time_limit)
        variables = self.define_variables(macs, model, x, y, iteration)
        is_feasible, y_loss, p_loss = self.compute_losses(macs, model, variables, x, y, iteration)

        # algorithm core: check for feasibility and behave depending on that
        if is_feasible:
            model.add(p_loss <= self.beta)
            model.minimize(y_loss)
        else:
            model.minimize(y_loss + (1.0 / self.alpha) * p_loss)

        # solve the problem and get the adjusted labels
        solution = model.solve()
        return self.return_solutions(macs, solution, variables, x, y, iteration)
