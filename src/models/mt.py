import time
import numpy as np

from moving_targets import MACS
from moving_targets.learners import Learner
from moving_targets.masters import CplexMaster
from moving_targets.metrics.constraints import MonotonicViolation
from src.models.model import Model


class MTLearner(Learner):
    def __init__(self, build_model, **kwargs):
        super(MTLearner, self).__init__()
        self.build_model = build_model
        self.fit_args = kwargs
        self.model = None

    def fit(self, macs, x, y, iteration, **kwargs):
        start_time = time.time()
        # re-initialize model
        self.model = self.build_model()
        # fit model
        fit_args = self.fit_args.copy()
        fit_args.update(kwargs)
        fit = self.model.fit(x[~np.isnan(y)], y[~np.isnan(y)], **fit_args)
        # log statistics
        macs.log(**{
            'time/learner': time.time() - start_time,
            'learner/epochs': fit.epoch[-1] + 1,
            'learner/loss': fit.history['loss'][-1]
        })

    def predict(self, x):
        return self.model.predict(x).reshape(-1, )


class MTMaster(CplexMaster):
    def __init__(self, monotonicities, augmented_mask, loss_fn='mse', alpha=1., learner_weights='all',
                 learner_omega=1.0, master_omega=None, eps=1e-3, time_limit=30):
        super(MTMaster, self).__init__(alpha=alpha, beta=1.0, time_limit=time_limit)
        assert loss_fn in ['mae', 'mse', 'sae', 'sse'], "Loss should be one in ['mae', 'mse', 'sae', 'sse']"
        self.higher_indices = np.array([hi for hi, _ in monotonicities])
        self.lower_indices = np.array([li for _, li in monotonicities])
        self.augmented_mask = augmented_mask
        self.loss_fn = getattr(CplexMaster, f'{loss_fn}_loss')
        self.learner_weights = learner_weights
        if learner_weights == 'all':
            self.infeasible_mask = np.ones_like(augmented_mask)
        elif learner_weights == 'infeasible':
            self.infeasible_mask = np.where(augmented_mask, False, True)
        else:
            raise ValueError("augmented_weights should be either 'all' or 'infeasible'")
        self.learner_omega = learner_omega
        self.master_omega = learner_omega if master_omega is None else master_omega
        self.eps = eps

    def build_model(self, macs, model, x, y, iteration):
        # handle 'projection' initial step (p = None)
        pred = None if not macs.fitted else macs.predict(x)
        # create variables and impose constraints for each monotonicity
        var = np.array(model.continuous_var_list(keys=len(y), name='y'))
        if len(self.higher_indices):
            model.add_constraints([h >= l for h, l in zip(var[self.higher_indices], var[self.lower_indices])])
        # return model info
        return var, pred

    def beta_step(self, macs, model, model_info, x, y, iteration):
        _, pred = model_info
        if len(self.higher_indices) == 0 or pred is None:
            violations = np.array([0])
        else:
            violations = pred[self.lower_indices] - pred[self.higher_indices]
            violations[violations < self.eps] = 0.0
        satisfied = violations == 0
        macs.log(**{
            'master/avg. violation': np.mean(violations),
            'master/pct. violation': 1 - np.mean(satisfied),
            'master/is feasible': int(satisfied.all())
        })
        return False

    def y_loss(self, macs, model, model_info, x, y, iteration):
        var, _ = model_info
        return self.loss_fn(model, y[~np.isnan(y)], var[~np.isnan(y)])

    def p_loss(self, macs, model, model_info, x, y, iteration):
        var, pred = model_info
        sw = np.where(self.augmented_mask, self.master_omega, 1)
        return 0.0 if pred is None else self.loss_fn(model, pred, var, sample_weight=sw)

    def return_solutions(self, macs, solution, model_info, x, y, iteration):
        var, pred = model_info
        adj = np.array([vy.solution_value for vy in var])
        mask = ~np.isnan(y)
        macs.log(**{
            'master/adj. mae': np.abs(adj[mask] - y[mask]).mean(),
            'master/adj. mse': np.mean((adj[mask] - y[mask]) ** 2),
            'time/master': solution.solve_details.time
        })
        if self.learner_weights == 'infeasible':
            diffs = pred[self.lower_indices] - pred[self.higher_indices]
            higher_mask = np.where(self.higher_indices[diffs > self.eps], True, False)
            lower_mask = np.where(self.lower_indices[diffs > self.eps], True, False)
            self.infeasible_mask = np.logical_or(self.infeasible_mask, np.logical_or(higher_mask, lower_mask))
        sample_weight = np.where(self.infeasible_mask, 1 / self.learner_omega, 0.0)
        sample_weight[~self.augmented_mask] = 1.0
        return adj, {'sample_weight': sample_weight}


class MT(MACS, Model):
    def __init__(self, learner, master, init_step='pretraining', metrics=None):
        super(MT, self).__init__(learner=learner, master=master, init_step=init_step, metrics=metrics)
        self.violation_metrics = [m for m in self.metrics if isinstance(m, MonotonicViolation)]
        self.metrics = [m for m in self.metrics if not isinstance(m, MonotonicViolation)]

    def on_iteration_end(self, macs, x, y, val_data, iteration):
        iteration = 0 if iteration == 'pretraining' else iteration
        logs = {'iteration': iteration, 'time/iteration': time.time() - self.time}
        # VIOLATION METRICS (on augmented data)
        p = self.predict(x)
        for violation_metric in self.violation_metrics:
            logs[f'metrics/{violation_metric.__name__}'] = violation_metric(x, y, p)
        # SCORE METRICS (on original data splits)
        for name, (xx, yy) in val_data.items():
            pp = self.predict(xx)
            for metric in self.metrics:
                logs[f'metrics/{name} {metric.__name__}'] = metric(xx, yy, pp)
        self.log(**logs)
