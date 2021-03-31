from docplex.mp.model import Model as CPModel

from src.moving_targets.masters import Master


class CplexMaster(Master):
    def __init__(self, alpha: float = 1., beta: float = 1., time_limit: float = 30.):
        super(CplexMaster, self).__init__(alpha, beta)
        self.time_limit = time_limit

    def build_model(self, macs, model, x, y, iteration):
        raise NotImplementedError("Please implement method 'build_model'")

    def is_feasible(self, macs, model, model_info, x, y, iteration):
        return True

    def y_loss(self, macs, model, model_info, x, y, iteration):
        return 0.0

    def p_loss(self, macs, model, model_info, x, y, iteration):
        return 0.0

    def return_solutions(self, macs, solution, model_info, x, y, iteration):
        return NotImplementedError("Please implement method 'return_solutions'")

    def adjust_targets(self, macs, x, y, iteration):
        # build model and get losses
        model = CPModel()
        model.set_time_limit(self.time_limit)
        model_info = self.build_model(macs, model, x, y, iteration)

        # algorithm core: check for feasibility and behave depending on that
        is_feasible = self.is_feasible(macs, model, model_info, x, y, iteration)
        p_loss = self.p_loss(macs, model, model_info, x, y, iteration)
        y_loss = self.y_loss(macs, model, model_info, x, y, iteration)
        if is_feasible:
            model.add(p_loss <= self.beta)
            model.minimize(y_loss)
        else:
            model.minimize(y_loss + (1.0 / self.alpha) * p_loss)

        # solve the problem and get the adjusted labels
        solution = model.solve()
        return self.return_solutions(macs, solution, model_info, x, y, iteration)
