import time
from typing import Any, Optional, Tuple, List

import numpy as np

from moving_targets import MACS
from moving_targets.learners import Learner
from moving_targets.masters import CplexMaster
from moving_targets.metrics import Metric
from moving_targets.metrics.constraints import MonotonicViolation
from src.models import MLP
from src.util.dictionaries import merge_dictionaries


class MTLearner(Learner):
    def __init__(self, loss, optimizer='adam', output_act=None, h_units=None, scalers=None, warm_start=False, **kwargs):
        super(MTLearner, self).__init__()
        self.output_act = output_act
        self.h_units = h_units
        self.scalers = scalers
        self.optimizer = optimizer
        self.loss = loss
        self.warm_start = warm_start
        self.fit_args = kwargs
        self.model = MLP(output_act=self.output_act, h_units=self.h_units, scalers=self.scalers)

    def fit(self, macs, x: Matrix, y: Vector, iteration, **kwargs):
        start_time = time.time()
        # re-initialize weights if warm start is not enabled, re-initialize optimizer in any case
        if not self.warm_start:
            self.model = MLP(output_act=self.output_act, h_units=self.h_units, scalers=self.scalers)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        # fit model
        fit_args = merge_dictionaries(self.fit_args, kwargs)
        fit = self.model.fit(x[~np.isnan(y)], y[~np.isnan(y)], **fit_args)
        # log statistics
        macs.log(**{
            'time/learner': time.time() - start_time,
            'learner/epochs': len(fit.epoch),
            'learner/loss': fit.history['loss'][-1] if len(fit.epoch) > 0 else np.nan
        })

    def predict(self, x):
        return self.model.predict(x).flatten()


class MTMaster(CplexMaster):
    def __init__(self, monotonicities: List[Tuple[int, int]], augmented_mask: np.ndarray, loss_fn: str,
                 alpha: float = 1.0, learner_weights: str = 'all', learner_omega: float = 1.0,
                 master_omega: Optional[float] = None, eps: float = 1e-3, time_limit: float = 30):
        super(MTMaster, self).__init__(alpha=alpha, beta=None, time_limit=time_limit)
        self.higher_indices = np.array([hi for hi, _ in monotonicities])
        self.lower_indices = np.array([li for _, li in monotonicities])
        self.augmented_mask = augmented_mask
        self.y_loss_fn = getattr(CplexMaster.losses, loss_fn[0] if isinstance(loss_fn, tuple) else loss_fn)
        self.p_loss_fn = getattr(CplexMaster.losses, loss_fn[1] if isinstance(loss_fn, tuple) else loss_fn)
        self.learner_weights = learner_weights
        if learner_weights == 'all':
            self.infeasible_mask = np.ones_like(augmented_mask)  # vector of 'True'
        elif learner_weights == 'infeasible':
            self.infeasible_mask = np.zeros_like(augmented_mask)  # vector of 'False'
        else:
            raise ValueError("learner_weights should be either 'all' or 'infeasible'")
        self.learner_omega = learner_omega
        if master_omega is None:
            self.master_omega_y, self.master_omega_p = learner_omega, learner_omega
        elif isinstance(master_omega, tuple):
            self.master_omega_y, self.master_omega_p = master_omega
        else:
            self.master_omega_y, self.master_omega_p = master_omega, master_omega
        self.eps = eps
        self.start_time = None

    def build_variables(self, model, y):
        raise NotImplementedError("Please implement method 'build variables'")

    # noinspection PyMethodMayBeStatic
    def build_predictions(self, macs, x):
        return macs.predict(x)

    def build_model(self, macs, model, x, y, iteration: Iteration):
        self.start_time = time.time()
        # handle 'projection' initial step (p = None)
        pred = None if not macs.fitted else self.build_predictions(macs, x)
        # create variables and impose constraints for each monotonicity
        var = np.array(self.build_variables(model, y))
        if len(self.higher_indices):
            model.add_constraints([h >= l for h, l in zip(var[self.higher_indices], var[self.lower_indices])])
        # return model info
        return var, pred

    def beta_step(self, macs, model, model_info, x, y, iteration: Iteration) -> bool:
        return False

    def y_loss(self, macs, model, model_info, x, y, iteration: Iteration) -> float:
        var, _ = model_info
        sw = np.where(self.augmented_mask, 1 / self.master_omega_y, 1)
        return self.y_loss_fn(model, y[~np.isnan(y)], var[~np.isnan(y)], sample_weight=sw)

    def p_loss(self, macs, model, model_info, x, y, iteration: Iteration) -> float:
        var, pred = model_info
        sw = np.where(self.augmented_mask, self.master_omega_p, 1)
        return 0.0 if pred is None else self.p_loss_fn(model, pred, var, sample_weight=sw)

    def return_solutions(self, macs, solution, model_info, x, y, iteration: Iteration) -> object:
        var, pred = model_info
        adj = np.array([vy.solution_value for vy in var])
        if self.learner_weights == 'infeasible':
            infeasible_mask = (pred[self.lower_indices] - pred[self.higher_indices]) > self.eps
            self.infeasible_mask[self.higher_indices[infeasible_mask]] = True
            self.infeasible_mask[self.lower_indices[infeasible_mask]] = True
        sample_weight = np.where(self.infeasible_mask, 1 / self.learner_omega, 0.0)
        sample_weight[~self.augmented_mask] = 1.0
        macs.log(**{'time/master': time.time() - self.start_time})
        self.start_time = None
        return adj, {'sample_weight': sample_weight}


class MTRegressionMaster(MTMaster):
    losses = {
        'mae': 'mean_absolute_error',
        'mean_absolute_error': 'mean_absolute_error',
        'mse': 'mean_squared_error',
        'mean_squared_error': 'mean_squared_error',
        'sae': 'sum_of_absolute_errors',
        'sum_of_absolute_errors': 'sum_of_absolute_errors',
        'sse': 'sum_of_squared_errors',
        'sum_of_squared_errors': 'sum_of_squared_errors'
    }

    def __init__(self,
                 monotonicities: List[Tuple[int, int]],
                 augmented_mask: np.ndarray,
                 loss_fn: str = 'mse',
                 **kwargs):
        assert loss_fn in self.losses.keys(), f'loss_fn should be in {list(self.losses.keys())}'
        super(MTRegressionMaster, self).__init__(
            monotonicities=monotonicities,
            augmented_mask=augmented_mask,
            loss_fn=self.losses[loss_fn],
            **kwargs
        )

    def build_variables(self, model, y):
        return model.continuous_var_list(keys=len(y), name='y', lb=-float('inf'), ub=float('inf'))

    def return_solutions(self, macs, solution, model_info, x, y, iteration: Iteration) -> object:
        adj, kwargs = super(MTRegressionMaster, self).return_solutions(macs, solution, model_info, x, y, iteration)
        mask = ~np.isnan(y)
        macs.log(**{
            'master/adj. mse': np.mean((adj[mask] - y[mask]) ** 2)
        })
        return adj, kwargs


class MTClassificationMaster(MTMaster):
    def __init__(self,
                 monotonicities: List[Tuple[int, int]],
                 augmented_mask: np.ndarray,
                 clip_value: float = 1e-15,
                 **kwargs):
        super(MTClassificationMaster, self).__init__(
            monotonicities=monotonicities,
            augmented_mask=augmented_mask,
            loss_fn='binary_crossentropy',
            **kwargs
        )
        # change clip values of binary crossentropy functions
        y_clip, p_clip = clip_value if isinstance(clip_value, tuple) else (clip_value, clip_value)
        assert y_clip > 0 and p_clip > 0, "clip_value should be either a positive number or a pair of positive numbers"
        self.y_loss_fn.clip_value = y_clip
        self.p_loss_fn.clip_value = p_clip

    def build_variables(self, model, y):
        return model.continuous_var_list(keys=len(y), name='y', lb=0.0, ub=1.0)

    def return_solutions(self, macs, solution, model_info, x, y, iteration: Iteration) -> object:
        adj, kwargs = super(MTClassificationMaster, self).return_solutions(macs, solution, model_info, x, y, iteration)
        mask = ~np.isnan(y)
        macs.log(**{
            'master/avg. flips': np.abs(adj[mask] - y[mask]).mean(),
            'master/tot. flips': np.abs(adj[mask] - y[mask]).sum(),
        })
        return adj, kwargs


class MT(MACS):
    def __init__(self, learner: MTLearner, master: MTMaster, init_step: str = 'pretraining',
                 metrics: Optional[List[Metric]] = None):
        super(MT, self).__init__(learner=learner, master=master, init_step=init_step, metrics=metrics)
        self.violation_metrics = [m for m in self.metrics if isinstance(m, MonotonicViolation)]
        self.metrics = [m for m in self.metrics if not isinstance(m, MonotonicViolation)]

    def on_iteration_end(self, macs, x: Matrix, y: Vector, val_data, iteration: Iteration, **kwargs):
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
