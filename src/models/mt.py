"""Moving Targets Models."""

import time
from typing import Optional, List, Tuple, Union, Any, Dict, Callable

import cvxpy as cp
import numpy as np
from gurobipy import GRB

from moving_targets import MACS
from moving_targets.learners import Learner
from moving_targets.masters import CplexMaster, GurobiMaster
from moving_targets.masters.cvxpy_master import CvxpyMaster
from moving_targets.metrics import Metric
from moving_targets.metrics.constraints import MonotonicViolation
from moving_targets.util.typing import Matrix, Vector, Iteration, MonotonicitiesList, Dataset
from src.models import MLP
from src.util.dictionaries import merge_dictionaries
from src.util.preprocessing import Scalers


class MTLearner(Learner):
    """Custom Moving Targets Learner."""

    def __init__(self,
                 loss: str,
                 optimizer: str = 'adam',
                 output_act: Optional[str] = None,
                 h_units: Optional[List[int]] = None,
                 scalers: Scalers = None,
                 warm_start: bool = False,
                 **fit_kwargs):
        """
        :param loss:
            The neural network loss function.

        :param optimizer:
            The neural network optimizer.

        :param output_act:
            The neural network output activation.

        :param h_units:
            The neural network hidden units.

        :param scalers:
            The x/y scalers.

        :param warm_start:
            Whether or not to use warm start during the moving targets iterations.

        :param fit_kwargs:
            Arguments for the `model.fit()` method (e.g., epochs, batch size, ...), which are combined to the keyword
            arguments passed to the `self.fit()` method.
        """
        super(MTLearner, self).__init__()

        self.output_act: Optional[str] = output_act
        """The neural network output activation."""

        self.h_units: Optional[List[int]] = h_units
        """The neural network hidden units."""

        self.scalers: Scalers = scalers
        """The x/y scalers."""

        self.optimizer: str = optimizer
        """The neural network optimizer."""

        self.loss: str = loss
        """The neural network loss function."""

        self.warm_start: bool = warm_start
        """Whether or not to use warm start during the moving targets iterations."""

        self.fit_args: Dict[str, Any] = fit_kwargs
        """Arguments for the `model.fit()` method (e.g., epochs, batch size, ...), which are combined to the keyword
        arguments passed to the `self.fit()` method."""

        self.model: MLP = MLP(output_act=self.output_act, h_units=self.h_units, scalers=self.scalers)
        """The neural model, which is an `MLP` instance."""

    def fit(self, macs, x: Matrix, y: Vector, iteration: Iteration, **fit_kwargs):
        """Fits the learner according to the implemented procedure using (x, y) as training data.

        :param macs:
            Reference to the `MACS` object encapsulating the learner.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param iteration:
            The current `MACS` iteration, usually a number.

        :param fit_kwargs:
            Arguments for the `model.fit()` method (e.g., epochs, batch size, ...) which are combined to the keyword
            arguments passed to the constructor.
        """
        start_time = time.time()
        # re-initialize weights if warm start is not enabled, re-initialize optimizer in any case
        if not self.warm_start:
            self.model = MLP(output_act=self.output_act, h_units=self.h_units, scalers=self.scalers)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        # fit model
        fit_args = merge_dictionaries(self.fit_args, fit_kwargs)
        fit = self.model.fit(x[~np.isnan(y)], y[~np.isnan(y)], **fit_args)
        # log statistics
        macs.log(**{
            'time/learner': time.time() - start_time,
            'learner/epochs': len(fit.epoch),
            'learner/loss': fit.history['loss'][-1] if len(fit.epoch) > 0 else np.nan
        })

    def predict(self, x):
        """Uses the fitted learner configuration to predict labels from input samples.

        :param x:
            The matrix/dataframe of input samples.

        :return:
            The vector of predicted labels.
        """
        return self.model.predict(x).flatten()


class MTMaster(CplexMaster, GurobiMaster, CvxpyMaster):
    """Custom Moving Targets Master Interface."""

    def __init__(self,
                 monotonicities: MonotonicitiesList,
                 augmented_mask: Vector,
                 loss_fn: str,
                 backend: str = 'cplex',
                 alpha: float = 1.0,
                 beta: Optional[float] = None,
                 learner_weights: str = 'all',
                 learner_omega: float = 1.0,
                 master_omega: Optional[float] = None,
                 eps: float = 1e-3,
                 **backend_kwargs):
        """
        :param monotonicities:
            List of monotonicities.

        :param augmented_mask:
            Boolean mask to distinguish between original and augmented samples.

        :param loss_fn:
            The master loss.

        :param backend:
            The solver to be used.

        :param alpha:
            The non-negative real number which is used to calibrate the two losses in the alpha step.

        :param beta:
            The non-negative real number which is used to constraint the p_loss in the beta step.

        :param learner_weights:
            Either 'all' or 'infeasible'.

        :param learner_omega:
            Real number that decides the weight of augmented samples during the learning step.

        :param master_omega:
            Real number that decides the weight of augmented samples during the master step.

        :param eps:
            The slack value under which a violation is considered to be acceptable.

        :param backend_kwargs:
            Any other specific argument to be passed to the super class (i.e., `CplexMaster`, `GurobiMaster`, or
            `CvxpyMaster` depending on the chosen backend).
        """
        assert eps >= 0, f"'eps' must be non-negative, but value {eps} was passed"
        if backend == 'cplex':
            super(MTMaster, self).__init__(alpha=alpha, beta=beta or 0.0, **backend_kwargs)
            losses = CplexMaster.losses
        elif backend == 'gurobi':
            super(CplexMaster, self).__init__(alpha=alpha, beta=beta or 0.0, **backend_kwargs)
            losses = GurobiMaster.losses
        elif backend == 'cvxpy':
            super(GurobiMaster, self).__init__(alpha=alpha, beta=beta or 0.0, **backend_kwargs)
            losses = CvxpyMaster.losses
        else:
            raise ValueError(f"'{backend}' is not a supported backend.")

        self.backend: str = backend
        """The solver to be used."""

        self.use_beta: bool = beta is not None
        """Whether or not to use the beta step during the Moving Targets process."""

        self.y_loss_fn: Callable = getattr(losses, loss_fn[0] if isinstance(loss_fn, tuple) else loss_fn)
        """The master loss respective to the ground truth."""

        self.p_loss_fn: Callable = getattr(losses, loss_fn[1] if isinstance(loss_fn, tuple) else loss_fn)
        """The master loss respective to the learner's predictions."""

        self.higher_indices: np.ndarray = np.array([hi for hi, _ in monotonicities])
        """The list of indices that are greater to the respective lower_indices."""

        self.lower_indices: np.ndarray = np.array([li for _, li in monotonicities])
        """The list of indices that are lower to the respective higher_indices."""

        self.augmented_mask: Vector = augmented_mask
        """Boolean mask to distinguish between original and augmented samples."""

        self.learner_weights: str = learner_weights
        """Either 'all' or 'infeasible'."""

        self.infeasible_mask = None
        """A vector of data points which have been part of a violation at least once during the process."""

        self.learner_omega = learner_omega
        """Real number that decides the weight of augmented samples during the learning step."""

        self.master_omega_y = None
        """Real number that decides the weight of the ground truth of augmented samples during the master step."""

        self.master_omega_p = None
        """Real number that decides the weight of the prediction of augmented samples during the master step."""

        self.eps: float = eps
        """The slack value under which a violation is considered to be acceptable."""

        self._start_time: Optional[float] = None
        """Internal variable to keep track of the elapsed time between iterations."""

        # handle infeasible mask depending on learner weights
        if learner_weights == 'all':
            self.infeasible_mask = np.ones_like(augmented_mask)  # vector of 'True'
        elif learner_weights == 'infeasible':
            self.infeasible_mask = np.zeros_like(augmented_mask)  # vector of 'False'
        else:
            raise ValueError(f"{learner_weights} is not a valid learner weights kind")

        # handle master omegas depending on master omega parameter
        if master_omega is None:
            self.master_omega_y, self.master_omega_p = learner_omega, learner_omega
        elif isinstance(master_omega, tuple):
            self.master_omega_y, self.master_omega_p = master_omega
        else:
            self.master_omega_y, self.master_omega_p = master_omega, master_omega

    def build_variables(self, model, y: Vector) -> Vector:
        """Template method which depends on the kind of task (classification vs. regression).

        :param model:
            The inner optimization model.

        :param y:
            The vector of training labels.

        :return:
            A vector of variables as in the solver (i.e., Cplex, Gurobi, Cvxpy).
        """
        raise NotImplementedError("Please implement method 'build_variables'")

    def build_predictions(self, macs, x: Matrix) -> Vector:
        """Returns the learner's predictions.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :return:
            The vector of predictions as floating point values.
        """
        return macs.predict(x)

    def build_model(self, macs, model, x: Matrix, y: Vector, iteration: Iteration) -> Any:
        """Creates the model variables leveraging the template method `self.build_variables()` then adds the monotonic
        constraints accordingly to the chosen solver backend.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param model:
            The inner optimization model.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param iteration:
            The current `MACS` iteration, usually a number (unused).

        :return:
            A tuple containing the vector of solver variables and the learner's predictions (None if not fitted yet).
        """
        self._start_time = time.time()
        # handle 'projection' initial step (p = None)
        pred = None if not macs.fitted else self.build_predictions(macs, x)
        # create variables and impose constraints for each monotonicity
        var = np.array(self.build_variables(model, y))
        if len(self.higher_indices):
            if self.backend == 'cplex':
                model.add_constraints([h >= l for h, l in zip(var[self.higher_indices], var[self.lower_indices])])
            elif self.backend == 'gurobi':
                model.addConstrs((h >= l for h, l in zip(var[self.higher_indices], var[self.lower_indices])), name='c')
            elif self.backend == 'cvxpy':
                model += [h >= l for h, l in zip(var[self.higher_indices], var[self.lower_indices])]
        # return model info
        return var, pred

    def beta_step(self, model_info: object, **compatibility_kwargs) -> bool:
        """Decides whether to use or not the beta step during the current iteration.

        :param model_info:
            The tuple (<variables>, <predictions>) returned by the 'build_model' function.

        :param compatibility_kwargs:
            Ignored additional arguments maintained for compatibility.

        :return:
            True if the beta step is allowed and the predictions do not violate any monotonicity, False otherwise.
        """
        _, pred = model_info
        if self.use_beta and pred is not None and len(self.higher_indices) > 0:
            return np.all(pred[self.lower_indices] - pred[self.higher_indices] <= self.eps)
        else:
            return False

    def y_loss(self, model, model_info, y: Vector, **compatibility_kwargs) -> Any:
        """Computes the loss of the model variables wrt real targets.

        :param model:
            The inner optimization model.

        :param model_info:
            The tuple (<variables>, <predictions>) returned by the 'build_model' function.

        :param y:
            The vector of training labels.

        :param compatibility_kwargs:
            Ignored additional arguments maintained for compatibility.

        :return:
            The y_loss, computed using the routine function `self.y_loss_fn`.
        """
        mask = ~np.isnan(y)
        var, _ = model_info
        sw = np.where(self.augmented_mask, 1 / self.master_omega_y, 1)
        return self.y_loss_fn(model, y[mask], var[mask], sample_weight=sw[mask])

    def p_loss(self, model, model_info, y: Vector, **compatibility_kwargs) -> Any:
        """Computes the loss of the model variables wrt predictions.

        :param model:
            The inner optimization model.

        :param model_info:
            The tuple (<variables>, <predictions>) returned by the 'build_model' function.

        :param y:
            The vector of training labels.

        :param compatibility_kwargs:
            Ignored additional arguments maintained for compatibility.

        :return:
            The y_loss, computed using the routine function `self.p_loss_fn`.
        """
        var, pred = model_info
        sw = np.where(self.augmented_mask, self.master_omega_p, 1)
        return 0.0 if pred is None else self.p_loss_fn(model, pred, var, sample_weight=sw)

    def return_solutions(self, macs, model_info, y: Vector, **compatibility_kwargs) -> Any:
        """Processes and returns the solutions given by the optimization model.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param model_info:
            The tuple (<variables>, <predictions>) returned by the 'build_model' function.

        :param y:
            The vector of training labels.

        :param compatibility_kwargs:
            Ignored additional arguments maintained for compatibility.

        :return:
            A tuple containing the adjusted labels and a dictionary of the form {'sample_weights': <sample_weights>}.
        """
        var, pred = model_info
        # adjusted targets
        adj = None
        if self.backend == 'cplex':
            adj = np.array([vy.solution_value for vy in var])
        elif self.backend == 'gurobi':
            adj = np.array([vy.x for vy in var])
        elif self.backend == 'cvxpy':
            adj = np.array([vy.value[0] for vy in var])
        # sample weights
        if self.learner_weights == 'infeasible' and pred is not None:
            infeasible_mask = (pred[self.lower_indices] - pred[self.higher_indices]) > self.eps
            self.infeasible_mask[self.higher_indices[infeasible_mask]] = True
            self.infeasible_mask[self.lower_indices[infeasible_mask]] = True
        sample_weight = np.ones(len(y)) if pred is None else np.where(self.infeasible_mask, 1 / self.learner_omega, 0)
        sample_weight[~self.augmented_mask] = 1.0
        # logs and outputs
        diffs = adj[~np.isnan(y)] - y[~np.isnan(y)]
        macs.log(**{
            'time/master': time.time() - self._start_time,
            'master/adj. mae': np.mean(np.abs(diffs)),
            'master/adj. mse': np.mean(diffs ** 2)
        })
        self._start_time = None
        return adj.astype(y.dtype), {'sample_weight': sample_weight}

    def adjust_targets(self, macs, x: Matrix, y: Vector, iteration: Iteration) -> Any:
        """Calls the correct super method accordingly to the chosen backend.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param iteration:
            The current `MACS` iteration, usually a number.

        :return:
            The output of the `self.return_solutions()` method.
        """
        if self.backend == 'cplex':
            return super(MTMaster, self).adjust_targets(macs, x, y, iteration)
        elif self.backend == 'gurobi':
            return super(CplexMaster, self).adjust_targets(macs, x, y, iteration)
        elif self.backend == 'cvxpy':
            return super(GurobiMaster, self).adjust_targets(macs, x, y, iteration)


class MTRegressionMaster(MTMaster):
    """Custom Moving Targets Master for Regression Problems. The static `losses` dictionary contains losses aliases."""

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
    """A dictionary of losses aliases."""

    def __init__(self,
                 monotonicities: MonotonicitiesList,
                 augmented_mask: Vector,
                 loss_fn: str = 'mse',
                 backend: str = 'cplex',
                 alpha: float = 1.0,
                 beta: Optional[float] = None,
                 learner_weights: str = 'all',
                 learner_omega: float = 1.0,
                 master_omega: Optional[float] = None,
                 eps: float = 1e-3,
                 **backend_kwargs):
        """
        :param monotonicities:
            List of monotonicities.

        :param augmented_mask:
            Boolean mask to distinguish between original and augmented samples.

        :param loss_fn:
            The master loss.

        :param backend:
            The solver to be used.

        :param alpha:
            The non-negative real number which is used to calibrate the two losses in the alpha step.

        :param beta:
            The non-negative real number which is used to constraint the p_loss in the beta step.

        :param learner_weights:
            Either 'all' or 'infeasible'.

        :param learner_omega:
            Real number that decides the weight of augmented samples during the learning step.

        :param master_omega:
            Real number that decides the weight of augmented samples during the master step.

        :param eps:
            The slack value under which a violation is considered to be acceptable.

        :param backend_kwargs:
            Any other specific argument to be passed to the super class (i.e., `CplexMaster`, `GurobiMaster`, or
            `CvxpyMaster` depending on the chosen backend).
        """
        assert loss_fn in self.losses.keys(), f"'{loss_fn}' is not a valid loss function"
        super(MTRegressionMaster, self).__init__(
            monotonicities=monotonicities,
            augmented_mask=augmented_mask,
            loss_fn=self.losses[loss_fn],
            backend=backend,
            alpha=alpha,
            beta=beta,
            learner_weights=learner_weights,
            learner_omega=learner_omega,
            master_omega=master_omega,
            eps=eps,
            **backend_kwargs
        )

    def build_variables(self, model, y: Vector) -> Vector:
        """Creates continuous model variables accordingly to the chosen backend.

        :param model:
            The inner optimization model.

        :param y:
            The vector of training labels.

        :return:
            A vector of variables as in the solver (i.e., Cplex, Gurobi, Cvxpy).
        """
        var = None
        if self.backend == 'cplex':
            var = model.continuous_var_list(keys=len(y), lb=-float('inf'), ub=float('inf'), name='y')
        elif self.backend == 'gurobi':
            var = model.addVars(len(y), vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'), name='y').values()
            model.update()
        elif self.backend == 'cvxpy':
            var = [cp.Variable((1,)) for _ in range(len(y))]
            for v in var:
                v.lb, v.ub = -float('inf'), float('inf')
        return var


class MTClassificationMaster(MTMaster):
    """Custom Moving Targets Master for Classification Problems. The static `losses` dictionary contains losses aliases,
    while the static `unsupported` dictionary contains the list of unsupported losses indexed by the given backend."""

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
    """A dictionary of losses aliases."""

    unsupported = {
        'cplex': ['reversed_binary_crossentropy', 'symmetric_binary_crossentropy'],
        'cvxpy': ['binary_hamming', 'binary_crossentropy', 'symmetric_binary_crossentropy']
    }
    """A dictionary of unsupported losses indexed via the backend solver."""

    def __init__(self,
                 monotonicities: MonotonicitiesList,
                 augmented_mask: Vector,
                 loss_fn: str = 'rbce',
                 backend: str = 'cplex',
                 alpha: float = 1.0,
                 beta: Optional[float] = None,
                 learner_weights: str = 'all',
                 learner_omega: float = 1.0,
                 master_omega: Optional[float] = None,
                 eps: float = 1e-3,
                 clip_value: Union[float, Tuple[float, float]] = 1e-3,
                 cont_eps: float = 1e-3,
                 **backend_kwargs):
        """
        :param monotonicities:
            List of monotonicities.

        :param augmented_mask:
            Boolean mask to distinguish between original and augmented samples.

        :param loss_fn:
            The master loss.

        :param backend:
            The solver to be used.

        :param alpha:
            The non-negative real number which is used to calibrate the two losses in the alpha step.

        :param beta:
            The non-negative real number which is used to constraint the p_loss in the beta step.

        :param learner_weights:
            Either 'all' or 'infeasible'.

        :param learner_omega:
            Real number that decides the weight of augmented samples during the learning step.

        :param master_omega:
            Real number that decides the weight of augmented samples during the master step.

        :param eps:
            The slack value under which a violation is considered to be acceptable.

        :param backend_kwargs:
            Any other specific argument to be passed to the super class (i.e., `CplexMaster`, `GurobiMaster`, or
            `CvxpyMaster` depending on the chosen backend).
        """
        # handle loss and build model from super class
        assert loss_fn in self.losses.keys(), f"'{loss_fn}' is not a valid loss function"
        self.cont_eps = cont_eps
        self.loss_fn = self.losses[loss_fn]
        super(MTClassificationMaster, self).__init__(
            monotonicities=monotonicities,
            augmented_mask=augmented_mask,
            loss_fn=self.losses[loss_fn],
            backend=backend,
            alpha=alpha,
            beta=beta,
            learner_weights=learner_weights,
            learner_omega=learner_omega,
            master_omega=master_omega,
            eps=eps,
            **backend_kwargs
        )
        # change clip values for crossentropy functions
        y_clip, p_clip = clip_value if isinstance(clip_value, tuple) else (clip_value, clip_value)
        assert y_clip > 0 and p_clip > 0, f"{clip_value} is not a valid clip value instance"
        self.y_loss_fn._clip_value = y_clip
        self.p_loss_fn._clip_value = p_clip
        # raise error if losses involving logarithms are called with cplex or binary variables with cvxpy
        for losses in MTClassificationMaster.unsupported.get(self.backend) or []:
            if self.loss_fn in losses:
                raise ValueError(f"{backend.capitalize()} Master cannot handle {self.loss_fn.replace('_', ' ')}")

    def build_variables(self, model, y: Vector) -> Vector:
        """Creates categorical model variables accordingly to the chosen backend.

        :param model:
            The inner optimization model.

        :param y:
            The vector of training labels.

        :return:
            A vector of variables as in the solver (i.e., Cplex, Gurobi, Cvxpy).
        """
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
        elif self.backend == 'cvxpy':
            var = [cp.Variable((1,)) for _ in range(len(y))]
            for v in var:
                v.lb, v.ub = self.cont_eps, 1 - self.cont_eps
                model += [v >= v.lb, v <= v.ub]
        return var

    def build_predictions(self, macs, x: Matrix) -> Vector:
        """Returns the predictions in the form of either classes (in case of hamming distance) or probabilities.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :return:
            The vector of predictions as floating point values.
        """
        p = macs.predict(x)
        if self.loss_fn in ['binary_hamming']:
            p = p.round().astype(int)
        return p


class MT(MACS):
    """Custom Model-Agnostic Constraint Satisfaction instance."""

    def __init__(self,
                 learner: MTLearner,
                 master: MTMaster,
                 init_step: str = 'pretraining',
                 metrics: Optional[List[Metric]] = None):
        """
        :param learner:
            A `MTLearner` instance.

        :param master:
            A `MTMaster` instance.

        :param init_step:
            The initial step of the algorithm, either 'pretraining' or 'projection'.

        :param metrics:
            A list of `Metric` instances to evaluate the final solution.
        """
        super(MT, self).__init__(learner=learner, master=master, init_step=init_step, metrics=metrics)
        self.violation_metrics = [m for m in self.metrics if isinstance(m, MonotonicViolation)]
        """Sublist of metrics involving constraint satisfaction."""

        self.metrics = [m for m in self.metrics if not isinstance(m, MonotonicViolation)]
        """Sublist of metrics involving prediction accuracy."""

    def on_iteration_end(self, x: Matrix, y: Vector, val_data: Dataset, iteration: Iteration, **compatibility_kwargs):
        """Logs the iteration, the elapsed time, the score metrics of the original splits (train, validation, test) and
        the constraint violation metrics on the augmented data.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

        :param iteration:
            The current `MACS` iteration, usually a number.

        :param compatibility_kwargs:
            Ignored additional arguments maintained for compatibility.
        """
        iteration = 0 if iteration == 'pretraining' else iteration
        logs = {'iteration': iteration, 'time/iteration': time.time() - self._time}
        # violation metrics (on augmented data)
        p = self.predict(x)
        for violation_metric in self.violation_metrics:
            logs[f'metrics/{violation_metric.__name__}'] = violation_metric(x, y, p)
        # score metrics (on original data splits)
        for name, (xx, yy) in val_data.items():
            pp = self.predict(xx)
            for metric in self.metrics:
                logs[f'metrics/{name} {metric.__name__}'] = metric(xx, yy, pp)
        self.log(**logs)
