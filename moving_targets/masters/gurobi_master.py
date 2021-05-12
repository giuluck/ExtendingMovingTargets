from abc import ABC

from gurobipy import Model, Env, GRB

from moving_targets.masters.losses import LossesHandler
from moving_targets.masters.master import Master


class GurobiMaster(Master, ABC):
    losses = LossesHandler(sum_fn=lambda model: sum, abs_fn=lambda model: abs, log_fn=lambda model: model.log)

    def __init__(self, alpha=1., beta=1., time_limit=30.):
        super(GurobiMaster, self).__init__(alpha=alpha, beta=beta)
        self.time_limit = time_limit

    def adjust_targets(self, macs, x, y, iteration):
        # build model and get losses
        with Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('TimeLimit', self.time_limit)
            env.start()
            with Model(env=env, name='model') as model:
                model_info = self.build_model(macs, model, x, y, iteration)
                model.update()

                # algorithm core: check for feasibility and behave depending on that
                y_loss = self.y_loss(macs, model, model_info, x, y, iteration)
                model.update()
                p_loss = self.p_loss(macs, model, model_info, x, y, iteration)
                model.update()
                if self.beta_step(macs, model, model_info, x, y, iteration):
                    model.addConstr(p_loss <= self.beta, name='loss')
                    model.update()
                    model.setObjective(y_loss, GRB.MINIMIZE)
                    model.update()
                else:
                    model.setObjective(y_loss + (1.0 / self.alpha) * p_loss, GRB.MINIMIZE)
                    model.update()

                # solve the problem and get the adjusted labels
                model.optimize()
                if model.Status == GRB.INFEASIBLE:
                    raise RuntimeError('The given model has no admissible solution, please check its constraints.')
                return self.return_solutions(macs, model, model_info, x, y, iteration)
