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
            'master/violations': violations / len(self.monotonicities),
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
    def __init__(self, learner: MTLearner, master: MTMaster, metrics=None, evaluation_data=None, **kwargs):
        super(MT, self).__init__(learner, master, init_step='pretraining', metrics=metrics)
        ar, nr, dr = np.meshgrid(np.linspace(1, 5, num=100), np.linspace(0, 200, num=100), ['D', 'DD', 'DDD', 'DDDD'])
        self.avg_ratings = ar.reshape(-1, )
        self.num_reviews = nr.reshape(-1, )
        self.dollar_ratings = dr.reshape(-1, )
        self.ground_truths = ctr_estimate(self.avg_ratings, self.num_reviews, self.dollar_ratings)
        self.evaluation_data = {} if evaluation_data is None else evaluation_data
        self.log(**kwargs)

    def on_pretraining_end(self, macs, x, y):
        self.on_iteration_end(macs, -1)

    def on_iteration_end(self, macs, idx):
        logs = {'iteration': idx + 1}
        for name, (xx, yy) in self.evaluation_data.items():
            pp = self.predict(xx)
            for metric in self.metrics:
                logs[f'learner/{name}_{metric.name}'] = metric(xx, yy, pp)
        pred = self.ctr_estimate(self.avg_ratings, self.num_reviews, self.dollar_ratings)
        logs['learner/ground_r2'] = r2_score(self.ground_truths, pred)
        self.log(**logs)
