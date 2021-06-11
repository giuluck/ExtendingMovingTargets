"""Moving Target Models."""

import time
from typing import Optional, List, Tuple, Union

import numpy as np
from gurobipy import GRB

from moving_targets import MACS
from moving_targets.learners import Learner
from moving_targets.masters import CplexMaster, GurobiMaster
from moving_targets.metrics import Metric
from moving_targets.metrics.constraints import MonotonicViolation
from moving_targets.util.typing import Matrix, Vector, Iteration, MonotonicitiesList
from src.models import MLP
from src.util.dictionaries import merge_dictionaries
from src.util.preprocessing import Scalers


class MTLearner(Learner):
    """Custom Moving Target Learner.

    Args:
        loss: the neural network loss function.
        optimizer: the neural network optimizer.
        output_act: the neural network output activation.
        h_units: the neural network hidden units.
        scalers: the x/y scalers.
        warm_start: whether or not to use warm start during the moving targets iterations..
        **kwargs: super-class arguments.
    """

    def __init__(self,
                 loss: str,
                 optimizer: str = 'adam',
                 output_act: Optional[str] = None,
                 h_units: Optional[List[int]] = None,
                 scalers: Scalers = None,
                 warm_start: bool = False,
                 **kwargs):
        super(MTLearner, self).__init__()
        self.output_act = output_act
        self.h_units = h_units
        self.scalers = scalers
        self.optimizer = optimizer
        self.loss = loss
        self.warm_start = warm_start
        self.fit_args = kwargs
        self.model = MLP(output_act=self.output_act, h_units=self.h_units, scalers=self.scalers)

    # noinspection PyMissingOrEmptyDocstring
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

    # noinspection PyMissingOrEmptyDocstring
    def predict(self, x):
        return self.model.predict(x).flatten()


class MTMaster(CplexMaster, GurobiMaster):
    """Custom Moving Target Master Interface.

    Args:
        monotonicities: list of monotonicities.
        augmented_mask: boolean mask to distinguish between original and augmented samples.
        loss_fn: the master loss.
        alpha: the non-negative real number which is used to calibrate the two losses in the alpha step.
        learner_weights: either 'all' or 'infeasible'.
        learner_omega: real number that decides the weight of augmented samples during the learning step.
        master_omega: real number that decides the weight of augmented samples during the master step.
        eps: the slack value under which a violation is considered to be acceptable.
        time_limit: the maximal time for which the master can run during each iteration.
        **kwargs: any other specific argument to be passed to the super class.
    """

    def __init__(self,
                 monotonicities: MonotonicitiesList,
                 augmented_mask: Vector,
                 loss_fn: str,
                 backend: str = 'cplex',
                 alpha: float = 1.0,
                 learner_weights: str = 'all',
                 learner_omega: float = 1.0,
                 master_omega: Optional[float] = None,
                 eps: float = 1e-3,
                 time_limit: Optional[float] = None,
                 **kwargs):
        self.backend: str = backend
        if self.backend == 'cplex':
            super(MTMaster, self).__init__(alpha=alpha, beta=0.0, time_limit=time_limit, **kwargs)
            self.y_loss_fn = getattr(CplexMaster.losses, loss_fn[0] if isinstance(loss_fn, tuple) else loss_fn)
            self.p_loss_fn = getattr(CplexMaster.losses, loss_fn[1] if isinstance(loss_fn, tuple) else loss_fn)
        elif self.backend == 'gurobi':
            super(CplexMaster, self).__init__(alpha=alpha, beta=0.0, time_limit=time_limit, **kwargs)
            self.y_loss_fn = getattr(GurobiMaster.losses, loss_fn[0] if isinstance(loss_fn, tuple) else loss_fn)
            self.p_loss_fn = getattr(GurobiMaster.losses, loss_fn[1] if isinstance(loss_fn, tuple) else loss_fn)
        else:
            raise ValueError(f"'{self.backend}' is not a supported backend.")
        self.higher_indices = np.array([hi for hi, _ in monotonicities])
        self.lower_indices = np.array([li for _, li in monotonicities])
        self.augmented_mask = augmented_mask
        self.learner_weights = learner_weights
        if learner_weights == 'all':
            self.infeasible_mask = np.ones_like(augmented_mask)  # vector of 'True'
        elif learner_weights == 'infeasible':
            self.infeasible_mask = np.zeros_like(augmented_mask)  # vector of 'False'
        else:
            raise ValueError(f"{learner_weights} is not a valid learner weights kind")
        self.learner_omega = learner_omega
        if master_omega is None:
            self.master_omega_y, self.master_omega_p = learner_omega, learner_omega
        elif isinstance(master_omega, tuple):
            self.master_omega_y, self.master_omega_p = master_omega
        else:
            self.master_omega_y, self.master_omega_p = master_omega, master_omega
        self.eps = eps
        self.start_time = None

    # noinspection PyMissingOrEmptyDocstring
    def build_variables(self, model, y) -> Vector:
        raise NotImplementedError("Please implement method 'build_variables'")

    # noinspection PyMissingOrEmptyDocstring, PyMethodMayBeStatic
    def build_predictions(self, macs, x):
        return macs.predict(x)

    # noinspection PyMissingOrEmptyDocstring
    def build_model(self, macs, model, x, y, iteration: Iteration):
        self.start_time = time.time()
        # handle 'projection' initial step (p = None)
        pred = None if not macs.fitted else self.build_predictions(macs, x)
        # create variables and impose constraints for each monotonicity
        var = np.array(self.build_variables(model, y))
        if len(self.higher_indices):
            if self.backend == 'cplex':
                model.add_constraints([h >= l for h, l in zip(var[self.higher_indices], var[self.lower_indices])])
            elif self.backend == 'gurobi':
                model.addConstrs((h >= l for h, l in zip(var[self.higher_indices], var[self.lower_indices])), name='c')
        # return model info
        return var, pred

    # noinspection PyMissingOrEmptyDocstring
    def beta_step(self, macs, model, model_info, x, y, iteration: Iteration) -> bool:
        return False

    # noinspection PyMissingOrEmptyDocstring
    def y_loss(self, macs, model, model_info, x, y, iteration: Iteration) -> float:
        var, _ = model_info
        sw = np.where(self.augmented_mask, 1 / self.master_omega_y, 1)
        return self.y_loss_fn(model, y[~np.isnan(y)], var[~np.isnan(y)], sample_weight=sw)

    # noinspection PyMissingOrEmptyDocstring
    def p_loss(self, macs, model, model_info, x, y, iteration: Iteration) -> float:
        var, pred = model_info
        sw = np.where(self.augmented_mask, self.master_omega_p, 1)
        return 0.0 if pred is None else self.p_loss_fn(model, pred, var, sample_weight=sw)

    # noinspection PyMissingOrEmptyDocstring
    def return_solutions(self, macs, solution, model_info, x, y, iteration: Iteration) -> object:
        var, pred = model_info
        # adjusted targets
        adj = None
        if self.backend == 'cplex':
            adj = np.array([vy.solution_value for vy in var])
        elif self.backend == 'gurobi':
            adj = np.array([vy.x for vy in var])
        # sample weights
        if self.learner_weights == 'infeasible':
            infeasible_mask = (pred[self.lower_indices] - pred[self.higher_indices]) > self.eps
            self.infeasible_mask[self.higher_indices[infeasible_mask]] = True
            self.infeasible_mask[self.lower_indices[infeasible_mask]] = True
        sample_weight = np.where(self.infeasible_mask, 1 / self.learner_omega, 0.0)
        sample_weight[~self.augmented_mask] = 1.0
        # logs and outputs
        diffs = adj[~np.isnan(y)] - y[~np.isnan(y)]
        macs.log(**{
            'time/master': time.time() - self.start_time,
            'master/adj. mae': np.mean(np.abs(diffs)),
            'master/adj. mse': np.mean(diffs ** 2)
        })
        self.start_time = None
        return adj.astype(y.dtype), {'sample_weight': sample_weight}

    # noinspection PyMissingOrEmptyDocstring
    def adjust_targets(self, macs, x: Matrix, y: Vector, iteration: Iteration) -> object:
        if self.backend == 'cplex':
            return super(MTMaster, self).adjust_targets(macs, x, y, iteration)
        elif self.backend == 'gurobi':
            return super(CplexMaster, self).adjust_targets(macs, x, y, iteration)


class MTRegressionMaster(MTMaster):
    """Custom Moving Target Master for Regression Problems.

    Args:
        monotonicities: list of monotonicities.
        augmented_mask: boolean mask to distinguish between original and augmented samples.
        loss_fn: the master loss.
        **kwargs: super-class arguments.
    """

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
                 monotonicities: MonotonicitiesList,
                 augmented_mask: Vector,
                 loss_fn: str = 'mse',
                 **kwargs):
        assert loss_fn in self.losses.keys(), f"'{loss_fn}' is not a valid loss function"
        super(MTRegressionMaster, self).__init__(
            monotonicities=monotonicities,
            augmented_mask=augmented_mask,
            loss_fn=self.losses[loss_fn],
            **kwargs
        )

    # noinspection PyMissingOrEmptyDocstring
    def build_variables(self, model, y) -> Vector:
        var = None
        if self.backend == 'cplex':
            var = model.continuous_var_list(keys=len(y), lb=-float('inf'), ub=float('inf'), name='y')
        elif self.backend == 'gurobi':
            var = model.addVars(len(y), vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name='y').values()
            model.update()
        return var


class MTClassificationMaster(MTMaster):
    """Custom Moving Target Master for Classification Problems.

    Args:
        monotonicities: list of monotonicities.
        augmented_mask: boolean mask to distinguish between original and augmented samples.
        loss_fn: the master loss.
        clip_value: the clipping value to be used to avoid numerical errors with the log in case of crossentropy.
        cont_eps: handles the lower/upper bounds for continuous variables in case of reversed/symmetric crossentropy.
        **kwargs: super-class arguments.
    """

    losses = {
        'hd': 'binary_hamming',
        'hamming_distance': 'binary_hamming',
        'binary_hamming': 'binary_hamming',
        'ce': 'binary_crossentropy',
        'crossentropy': 'binary_crossentropy',
        'bce': 'binary_crossentropy',
        'binary crossentropy': 'binary_crossentropy',
        'rce': 'reversed_binary_crossentropy',
        'reversed crossentropy': 'reversed_binary_crossentropy',
        'rbce': 'reversed_binary_crossentropy',
        'reversed binary crossentropy': 'reversed_binary_crossentropy',
        'sce': 'symmetric_binary_crossentropy',
        'symmetric crossentropy': 'symmetric_binary_crossentropy',
        'sbce': 'symmetric_binary_crossentropy',
        'symmetric binary crossentropy': 'symmetric_binary_crossentropy'
    }

    def __init__(self,
                 monotonicities: MonotonicitiesList,
                 augmented_mask: Vector,
                 loss_fn: str = 'hd',
                 clip_value: Union[float, Tuple[float, float]] = 1e-3,
                 cont_eps: float = 1e-3,
                 **kwargs):
        # handle loss and build model from super class
        assert loss_fn in self.losses.keys(), f"'{loss_fn}' is not a valid loss function"
        self.cont_eps = cont_eps
        self.loss_fn = self.losses[loss_fn]
        super(MTClassificationMaster, self).__init__(
            monotonicities=monotonicities,
            augmented_mask=augmented_mask,
            loss_fn=self.losses[loss_fn],
            **kwargs
        )
        # change clip values for crossentropy functions
        y_clip, p_clip = clip_value if isinstance(clip_value, tuple) else (clip_value, clip_value)
        assert y_clip > 0 and p_clip > 0, f"{clip_value} is not a valid clip value instance"
        self.y_loss_fn.clip_value = y_clip
        self.p_loss_fn.clip_value = p_clip
        # raise error if losses involving logarithms are called with cplex
        if self.backend == 'cplex' and ('reversed' in self.loss_fn or 'symmetric' in self.loss_fn):
            raise ValueError(f"Cplex Master cannot handle {self.loss_fn.replace('_', ' ')}")

    # noinspection PyMissingOrEmptyDocstring
    def build_variables(self, model, y) -> Vector:
        var = None
        if self.backend == 'cplex':
            var = model.binary_var_list(keys=len(y), lb=0.0, ub=1.0, name='y')
        elif self.backend == 'gurobi':
            if self.loss_fn in ['binary_hamming', 'binary_crossentropy']:
                var = model.addVars(len(y), vtype=GRB.BINARY, lb=0.0, ub=1.0, name='y').values()
            else:
                lb, ub = self.cont_eps, 1 - self.cont_eps
                var = model.addVars(len(y), vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name='y').values()
            model.update()
        return var

    # noinspection PyMissingOrEmptyDocstring
    def build_predictions(self, macs, x):
        p = macs.predict(x)
        if self.loss_fn in ['binary_hamming']:
            p = p.round().astype(int)
        return p

    # noinspection PyMissingOrEmptyDocstring
    def y_loss(self, macs, model, model_info, x, y, iteration: Iteration) -> float:
        m = ~np.isnan(y)
        v, p = model_info
        # masks both y variables and model variables (v) to transform the vector to type int in order to deal with loss
        return super(MTClassificationMaster, self).y_loss(macs, model, (v[m], p), x, y[m].astype(int), iteration)


class MT(MACS):
    """Custom Model-Agnostic Constraint Satisfaction instance.

    Args:
        learner: a `MTLearner` instance.
        master: a `MTMaster` instance.
        init_step: the initial step of the algorithm, either 'pretraining' or 'projection'.
        metrics: a list of `Metric` instances to evaluate the final solution.
    """

    def __init__(self,
                 learner: MTLearner,
                 master: MTMaster,
                 init_step: str = 'pretraining',
                 metrics: Optional[List[Metric]] = None):
        super(MT, self).__init__(learner=learner, master=master, init_step=init_step, metrics=metrics)
        self.violation_metrics = [m for m in self.metrics if isinstance(m, MonotonicViolation)]
        self.metrics = [m for m in self.metrics if not isinstance(m, MonotonicViolation)]

    # noinspection PyMissingOrEmptyDocstring
    def on_iteration_end(self, macs, x: Matrix, y: Vector, val_data, iteration: Iteration, **kwargs):
        iteration = 0 if iteration == 'pretraining' else iteration
        logs = {'iteration': iteration, 'time/iteration': time.time() - self.time}
        # validation metrics (on augmented data)
        p = self.predict(x)
        for violation_metric in self.violation_metrics:
            logs[f'metrics/{violation_metric.__name__}'] = violation_metric(x, y, p)
        # score metrics (on original data splits)
        for name, (xx, yy) in val_data.items():
            pp = self.predict(xx)
            for metric in self.metrics:
                logs[f'metrics/{name} {metric.__name__}'] = metric(xx, yy, pp)
        self.log(**logs)
