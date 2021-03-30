import time
import numpy as np
from sklearn.metrics import r2_score

from src.moving_targets.learners import Learner
from src.moving_targets.masters import CplexMaster
from src.moving_targets import MACS
from src.restaurants.models import Model as RestaurantModel
from src.restaurants import ctr_estimate


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
        self.monotonicities = monotonicities

    def define_variables(self, macs, model, x, y, iteration):
        return model.continuous_var_list(keys=len(y), lb=0.0, ub=1.0, name='y')

    def compute_losses(self, macs, model, variables, x, y, iteration):
        assert macs.init_step == 'pretraining', "This master supports 'pretraining' initial step only"
        p = macs.predict(x)

        # for each monotonicity, add the constraint and change the feasibility based on whether that is satisfied or not
        violations = 0
        for higher, lower in self.monotonicities:
            violations += 0 if p[higher] >= p[lower] else 1
            model.add_constraint(variables[higher] >= variables[lower])

        # define the total loss w.r.t. the true labels (y) and the loss w.r.t. the predictions (p)
        y_loss = [model.abs(yv - cpv) for yv, cpv in zip(y, variables) if yv != -1]
        y_loss = model.sum(y_loss) / len(y_loss)
        p_loss = [model.abs(pv - cpv) for pv, cpv in zip(p, variables)]
        p_loss = model.sum(p_loss) / len(p_loss)

        macs.log(**{
            'master/violations': 0 if len(self.monotonicities) == 0 else violations / len(self.monotonicities),
            'master/feasible': 1 if violations == 0 else 0
        })
        return violations == 0, y_loss, p_loss

    def return_solutions(self, macs, solution, variables, x, y, iteration):
        adj_y = np.array([vy.solution_value for vy in variables])
        macs.log(**{
            'master/adjusted mse': np.mean((adj_y[y != -1] - y[y != -1]) ** 2),
            'master/mip gap': solution.solve_details.gap,
            'time/master': solution.solve_details.time
        })
        return adj_y


class MT(MACS, RestaurantModel):
    def __init__(self, learner: MTLearner, master: MTMaster, metrics=None, evaluation_data=None):
        super(MT, self).__init__(learner, master, init_step='pretraining', metrics=metrics)
        self.evaluation_data = {} if evaluation_data is None else evaluation_data

    def on_pretraining_end(self, macs, x, y):
        self.on_iteration_end(macs, -1)

    def on_iteration_end(self, macs, idx):
        logs = {'iteration': idx + 1}
        for name, (xx, yy) in self.evaluation_data.items():
            pp = self.predict(xx)
            for metric in self.metrics:
                logs[f'learner/{name}_{metric.name}'] = metric(xx, yy, pp)
        logs['learner/ground_r2'] = self.compute_ground_r2()
        self.log(**logs)
