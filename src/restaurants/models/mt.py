import time
import numpy as np

from src.moving_targets.learners import Learner
from src.moving_targets.masters import CplexMaster
from src.moving_targets import MACS
from src.restaurants.models import Model as RestaurantModel


class MTLearner(Learner):
    def __init__(self, build_model, restart_fit=True, **kwargs):
        super(MTLearner, self).__init__()
        self.build_model = build_model
        self.restart_fit = restart_fit
        self.fit_args = kwargs
        self.model = build_model()

    def fit(self, macs, x, y, iteration, **kwargs):
        start_time = time.time()
        if self.restart_fit:
            self.model = self.build_model()
        self.model.fit(x[y != -1], y[y != -1], **self.fit_args, **kwargs)
        macs.log(**{'time/learner': time.time() - start_time})

    def predict(self, x):
        return self.model.predict(x).reshape(-1, )


class MTMaster(CplexMaster):
    def __init__(self, monotonicities, alpha=1., beta=1., time_limit=30):
        super(MTMaster, self).__init__(alpha=alpha, beta=beta, time_limit=time_limit)
        self.higher_indices = np.array([hi for hi, _ in monotonicities])
        self.lower_indices = np.array([li for _, li in monotonicities])

    def build_model(self, macs, model, x, y, iteration):
        # handle 'projection' initial step (p = None)
        p = None if iteration == 0 and macs.init_step == 'projection' else macs.predict(x)
        # create variables and impose constraints for each monotonicity
        variables = np.array(model.continuous_var_list(keys=len(y), lb=0.0, ub=1.0, name='y'))
        model.add_constraints([h >= l for h, l in zip(variables[self.higher_indices], variables[self.lower_indices])])
        # return model info
        return variables, p

    def is_feasible(self, macs, model, model_info, x, y, iteration):
        variables, p = model_info
        if len(self.higher_indices) == 0 or p is None:
            violations = 0
        else:
            violations = np.mean(p[self.higher_indices] < p[self.lower_indices])
        macs.log(**{
            'master/violations': violations,
            'master/feasible': int(violations == 0)
        })
        return violations == 0

    def y_loss(self, macs, model, model_info, x, y, iteration):
        variables, _ = model_info
        y_loss = [model.abs(yy - vv) for yy, vv in zip(y[y != -1], variables[y != -1])]
        return model.sum(y_loss) / len(y_loss)

    def p_loss(self, macs, model, model_info, x, y, iteration):
        variables, p = model_info
        if p is None:
            return 0.0
        else:
            return model.sum([model.abs(pp - vv) for pp, vv in zip(p, variables)]) / len(p)

    def return_solutions(self, macs, solution, model_info, x, y, iteration):
        variables, _ = model_info
        adj_y = np.array([vy.solution_value for vy in variables])
        macs.log(**{
            'master/adjusted mse': np.mean((adj_y[y != -1] - y[y != -1]) ** 2),
            'master/mip gap': solution.solve_details.gap,
            'time/master': solution.solve_details.time
        })
        # TODO: compute sample weights and return adjusted labels and sample weights
        return adj_y


class MT(MACS, RestaurantModel):
    def __init__(self, learner: MTLearner, master: MTMaster, init_step='pretraining', metrics=None):
        super(MT, self).__init__(learner, master, init_step=init_step, metrics=metrics)

    def on_pretraining_end(self, macs, x, y, val_data):
        self.on_iteration_end(macs, x, y, val_data, -1)

    def on_iteration_end(self, macs, x, y, val_data, iteration):
        logs = {'iteration': iteration + 1}
        for name, (xx, yy) in val_data.items():
            pp = self.predict(xx)
            for metric in self.metrics:
                logs[f'learner/{name}_{metric.name}'] = metric(xx, yy, pp)
        logs['learner/ground_r2'] = self.compute_ground_r2()
        self.log(**logs)
