import time
import numpy as np

from moving_targets import MACS
from moving_targets.learners import Learner
from moving_targets.masters import CplexMaster
from src.models.model import Model


class MTLearner(Learner):
    def __init__(self, build_model, warm_start=False, **kwargs):
        super(MTLearner, self).__init__()
        self.build_model = build_model
        self.warm_start = warm_start
        self.fit_args = kwargs
        self.model = build_model()

    def fit(self, macs, x, y, iteration, sample_weight=None, **kwargs):
        start_time = time.time()
        # re-instantiate model if no warm start
        if not self.warm_start:
            self.model = self.build_model()
        # mask values with nan label or sample_weight == 0.0 and fit the model
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        mask = ~np.logical_or(sample_weight == 0.0, np.isnan(y))
        fit = self.model.fit(x[mask], y[mask], **self.fit_args, sample_weight=sample_weight[mask], **kwargs)
        # retrieve number of epochs and last loss depending on the model
        if 'keras' in str(type(fit)):
            epochs, loss = fit.epoch[-1] + 1, fit.history['loss'][-1]
        elif 'sklearn' in str(type(fit)):
            epochs, loss = fit.n_iter_, fit.loss_
        else:
            epochs, loss = np.nan, np.nan
        # log info
        macs.log(**{
            'time/learner': time.time() - start_time,
            'learner/epochs': epochs,
            'learner/loss': loss
        })

    def predict(self, x):
        return self.model.predict(x).reshape(-1, )


class MTMaster(CplexMaster):
    def __init__(self, monotonicities, loss_fn='mae', alpha=1., beta=1., time_limit=30):
        super(MTMaster, self).__init__(alpha=alpha, beta=beta, time_limit=time_limit)
        assert loss_fn in ['mae', 'mse', 'sae', 'sse'], "Loss should be one in ['mae', 'mse', 'sae', 'sse']"
        self.loss_fn = getattr(CplexMaster, f'{loss_fn}_loss')
        self.higher_indices = np.array([hi for hi, _ in monotonicities])
        self.lower_indices = np.array([li for _, li in monotonicities])

    def build_model(self, macs, model, x, y, iteration):
        # handle 'projection' initial step (p = None)
        pred = None if not macs.fitted else macs.predict(x)
        # create variables and impose constraints for each monotonicity
        var = np.array(model.continuous_var_list(keys=len(y), name='y'))
        if len(self.higher_indices):
            model.add_constraints([h >= l for h, l in zip(var[self.higher_indices], var[self.lower_indices])])
        # return model info
        return var, pred

    def is_feasible(self, macs, model, model_info, x, y, iteration):
        _, pred = model_info
        if len(self.higher_indices) == 0 or pred is None:
            violations = np.array([0])
        else:
            violations = np.maximum(0.0, pred[self.lower_indices] - pred[self.higher_indices])
        satisfied = violations == 0
        macs.log(**{
            'master/avg. violation': np.mean(violations),
            'master/pct. violation': 1 - np.mean(satisfied),
            'master/is feasible': int(satisfied.all())
        })
        return satisfied.all()

    def y_loss(self, macs, model, model_info, x, y, iteration):
        var, _ = model_info
        mask = ~np.isnan(y)
        return self.loss_fn(model, y[mask], var[mask])

    def p_loss(self, macs, model, model_info, x, y, iteration):
        var, pred = model_info
        return 0.0 if pred is None else self.loss_fn(model, pred, var)

    def return_solutions(self, macs, solution, model_info, x, y, iteration):
        var, _ = model_info
        adj = np.array([vy.solution_value for vy in var])
        mask = ~np.isnan(y)
        macs.log(**{
            'master/adj. mae': np.abs(adj[mask] - y[mask]).mean(),
            'master/adj. mse': np.mean((adj[mask] - y[mask]) ** 2),
            'time/master': solution.solve_details.time
        })
        return adj


class MT(MACS, Model):
    def __init__(self, learner, master, init_step='pretraining', metrics=None):
        super(MT, self).__init__(learner=learner, master=master, init_step=init_step, metrics=metrics)

    def on_iteration_end(self, macs, x, y, val_data, iteration):
        iteration = 0 if iteration == 'pretraining' else iteration
        logs = {'iteration': iteration, 'time/iteration': time.time() - self.time}
        for name, (xx, yy) in val_data.items():
            pp = self.predict(xx)
            for metric in self.metrics:
                logs[f'metrics/{name}_{metric.__name__}'] = metric(xx, yy, pp)
        self.log(**logs)
