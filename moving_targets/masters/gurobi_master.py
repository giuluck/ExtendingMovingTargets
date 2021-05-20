from abc import ABC

from gurobipy import Model, Env, GRB

from moving_targets.masters.losses import LossesHandler
from moving_targets.masters.master import Master


def _abs(model, x):
    abs_x = model.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name=f'abs({x})')
    aux_x = model.addVar(lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name=f'aux({x})')
    model.update()
    model.addConstr(aux_x == x)
    model.addGenConstrAbs(abs_x, aux_x, name=f'abs({x})')
    return abs_x


def _log(model, x):
    log_x = model.addVar(lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name=f'abs({x})')
    aux_x = model.addVar(lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name=f'aux({x})')
    model.update()
    model.addConstr(aux_x == x)
    model.addGenConstrLog(log_x, aux_x, name=f'abs({x})')
    return log_x


class GurobiMaster(Master, ABC):
    losses = LossesHandler(sum_fn=lambda model, x: sum(x), abs_fn=_abs, log_fn=_log)

    def __init__(self, alpha=1., beta=1., time_limit=30.):
        super(GurobiMaster, self).__init__(alpha=alpha, beta=beta)
        self.time_limit = time_limit

    def adjust_targets(self, macs, x, y, iteration):
        # build model and get losses
        with Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with Model(env=env, name='model') as model:
                model.setParam('TimeLimit', self.time_limit)
                model_info = self.build_model(macs, model, x, y, iteration)
                # algorithm core: check for feasibility and behave depending on that
                y_loss = self.y_loss(macs, model, model_info, x, y, iteration)
                p_loss = self.p_loss(macs, model, model_info, x, y, iteration)
                if self.beta_step(macs, model, model_info, x, y, iteration):
                    model.addConstr(p_loss <= self.beta, name='loss')
                    model.setObjective(y_loss, GRB.MINIMIZE)
                else:
                    model.setObjective(y_loss + (1.0 / self.alpha) * p_loss, GRB.MINIMIZE)

                # solve the problem and get the adjusted labels
                model.optimize()
                if model.Status == GRB.INFEASIBLE:
                    raise RuntimeError('The given model has no admissible solution, please check its constraints.')
                return self.return_solutions(macs, model, model_info, x, y, iteration)
