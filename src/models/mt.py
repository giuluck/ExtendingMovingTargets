"""Moving Targets Models."""
from typing import Optional, List, Any, Dict

import numpy as np
from moving_targets import MACS
from moving_targets.learners import Learner
from moving_targets.masters import Master
from moving_targets.masters.backends import GurobiBackend
from moving_targets.masters.losses import Loss, aliases, HammingDistance, CrossEntropy, MSE, SAE, SSE, MAE
from moving_targets.masters.optimizers import Optimizer, BetaClassSatisfiability
from moving_targets.metrics import Metric
from moving_targets.metrics.constraints import MonotonicViolation
from moving_targets.util.typing import Dataset

from src.models import MLP
from src.util.preprocessing import Scalers
from src.util.typing import MonotonicitiesList


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
        super(MTLearner, self).__init__(stats=True)

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

    def fit(self, x, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> Any:
        # re-initialize weights if warm start is not enabled, re-initialize optimizer in any case
        if not self.warm_start:
            self.model = MLP(output_act=self.output_act, h_units=self.h_units, scalers=self.scalers)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        # fit model
        fit = self.model.fit(x[~np.isnan(y)], y[~np.isnan(y)], **self.fit_args)
        # log statistics
        self._log_stats(epochs=len(fit.epoch), loss=fit.history['loss'][-1] if len(fit.epoch) > 0 else np.nan)
        return self

    def predict(self, x) -> np.ndarray:
        return self.model.predict(x).flatten()


class MTMaster(Master):
    """Custom Moving Targets Master."""

    # TODO: remove unused parameters kept for compatibility
    def __init__(self,
                 monotonicities: MonotonicitiesList,
                 augmented_mask: np.ndarray,
                 classification: bool,
                 y_loss: str = 'mse',
                 p_loss: str = 'mse',
                 alpha: float = 1.0,
                 beta: Optional[float] = None,
                 learner_weights: str = 'all',
                 learner_omega: float = 1.0,
                 master_omega: Optional[float] = None,
                 eps: float = 1e-3):
        """
        :param monotonicities:
            List of monotonicities.

        :param augmented_mask:
            Boolean mask to distinguish between original and augmented samples.

        :param classification:
            Whether the master should solve a (binary) classification or a regression task.

        :param y_loss:
            The master y_loss.

        :param p_loss:
            The master p_loss.

        :param alpha:
            The non-negative real number which is used to calibrate the two losses in the alpha step.

        :param beta:
            The non-negative real number which is used to constraint the p_loss in the beta step.

        :param eps:
            The slack value under which a violation is considered to be acceptable.
        """
        y_class, p_class = aliases.get(y_loss), aliases.get(p_loss)
        assert y_class is not None, f"Unknown y_loss '{y_loss}'"
        assert p_class is not None, f"Unknown p_loss '{p_loss}'"
        if classification:
            # use binary targets only if required (i.e., HammingDistance or CrossEntropy loss), otherwise continuous
            binary = y_class in [HammingDistance, CrossEntropy] or p_class in [HammingDistance, CrossEntropy]
            y_loss = y_class() if y_class in [HammingDistance, CrossEntropy] else y_class(binary=binary)
            p_loss = p_class() if p_class in [HammingDistance, CrossEntropy] else p_class(binary=binary)
            beta = None if beta is None else BetaClassSatisfiability(base=beta)
            lb, ub, vtype = 0, 1, 'binary' if binary else 'continuous'
        else:
            # check that the loss is a valid regression loss
            assert y_class in [SAE, SSE, MAE, MSE], f"Unsupported y_loss '{y_loss}'"
            assert p_class in [SAE, SSE, MAE, MSE], f"Unsupported p_loss '{p_loss}'"
            y_loss, p_loss = y_class(), p_class()
            beta = None if beta is None else Optimizer(base=beta)
            lb, ub, vtype = -float('inf'), float('inf'), 'continuous'

        class MaskedLoss(Loss):
            """Custom loss wrapping another loss object that is used to ignore augmented samples in the y_loss."""

            def __init__(self, loss: Loss, mask: np.ndarray):
                super(MaskedLoss, self).__init__(name=loss.__name__)
                self.loss, self.mask = loss, mask

            def __call__(self, backend, numeric_variables, model_variables, sample_weight=None):
                return self.loss(backend, numeric_variables[self.mask], model_variables[self.mask], sample_weight)

        super(MTMaster, self).__init__(backend=GurobiBackend(time_limit=30),
                                       alpha=Optimizer(base=alpha),
                                       beta=beta,
                                       y_loss=MaskedLoss(loss=y_loss, mask=~augmented_mask),
                                       p_loss=p_loss,
                                       stats=True)

        self.lb = lb
        """The variables lower bounds."""

        self.ub = ub
        """The variables upper bounds."""

        self.vtype = vtype
        """The variables vtypes."""

        self.higher_indices: np.ndarray = np.array([hi for hi, _ in monotonicities])
        """The list of indices that are greater to the respective lower_indices."""

        self.lower_indices: np.ndarray = np.array([li for _, li in monotonicities])
        """The list of indices that are lower to the respective higher_indices."""

        self.eps: float = eps
        """The slack value under which a violation is considered to be acceptable."""

    def use_beta(self, x, y: np.ndarray, p: np.ndarray) -> bool:
        return np.all(p[self.lower_indices] - p[self.higher_indices] <= self.eps)

    def build(self, x, y: np.ndarray) -> np.ndarray:
        v = self.backend.add_variables(len(y), vtype=self.vtype, lb=self.lb, ub=self.ub, name='y')
        self.backend.add_constraints([h >= l for h, l in zip(v[self.higher_indices], v[self.lower_indices])])
        return v


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
        self.train_metrics: List[Metric] = [m for m in metrics if isinstance(m, MonotonicViolation)]
        """Sublist of metrics involving monotonicities which should be evaluated on augmented (train) data only."""

        self.val_metrics: List[Metric] = [m for m in metrics if not isinstance(m, MonotonicViolation)]
        """Sublist of metrics not involving monotonicities which should be evaluated on validation data only."""

        super(MT, self).__init__(learner=learner, master=master, init_step=init_step, metrics=self.train_metrics)

    def on_iteration_end(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        # compute metrics on validation data using validation metrics, then go back to train metrics
        self.metrics = self.val_metrics
        super(MT, self).on_iteration_end(macs=macs, x=x, y=y, val_data=val_data)
        self.metrics = self.train_metrics
