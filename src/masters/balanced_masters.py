"""Master implementation for the Balance Counts problem."""
from collections import namedtuple
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from moving_targets.learners.learner import Classifier
from moving_targets.masters import LossesHandler
from moving_targets.masters.cplex_master import CplexMaster
from moving_targets.util.typing import Iteration


class BalancedCounts(CplexMaster):
    """Master for the Balanced Counts problem in which output classes are constrained to be equally distributed."""

    Info = namedtuple('Info', 'variables predictions max_count')
    """Data structure for the model info returned by the 'build_model()' method."""

    losses: LossesHandler = LossesHandler(sum_fn=lambda m, v: m.sum(v.flatten()),
                                          abs_fn=lambda m, mvs, nvs: nvs * (1 - mvs) + (1 - nvs) * mvs,
                                          log_fn=None)
    """The `LossesHandler` object for this backend solver.
    
    Uses a custom absolute function that speeds up the computation due to the assumption of binary model variables
    (i.e., it computes an hamming distance with continuous numeric targets).
    """

    accepted_losses: Dict[str, str] = {
        'hd': 'categorical_hamming',
        'hamming_distance': 'categorical_hamming',
        'ce': 'categorical_crossentropy',
        'crossentropy': 'categorical_crossentropy',
        'mae': 'mean_absolute_error',
        'mean_absolute_error': 'mean_absolute_error',
        'mse': 'mean_squared_error',
        'mean_squared_error': 'mean_squared_error'
    }
    """A dictionary of losses aliases that are accepted."""

    def __init__(self, loss_fn: str = 'hd', clip_value: float = 1e-2, slack: float = 1.05, alpha: float = 1.,
                 beta: float = 1., time_limit: Optional[float] = 30):
        """
        :param loss_fn:
            The loss function computed between the model variables and the learner predictions.

            Must be one in ['hamming_distance', 'crossentropy', 'mean_squared_error' and 'mean_absolute_error'].

        :param clip_value:
            The clipping value for predicted probabilities.

        :param slack:
            The slack value, expressed in terms of ratio wrt the other classes, to allow some output class to have a
            few labelled samples more than the other classes.

        :param alpha:
            The non-negative real number which is used to calibrate the two losses in the alpha step.

        :param beta:
            The non-negative real number which is used to constraint the p_loss in the beta step.

        :param time_limit:
            The maximal time for which the master can run during each iteration.
        """
        assert loss_fn in self.accepted_losses.keys(), f"'{loss_fn}' is not a valid loss function"
        super(BalancedCounts, self).__init__(alpha=alpha, beta=beta, time_limit=time_limit)

        self.clip_value: float = clip_value
        """The clipping value for probabilities, in case they are used."""

        self.slack: float = slack
        """The slack value, expressed in terms of ratio wrt the other classes, to allow some output class to have a
        few labelled samples more than the other classes."""

        self._p_loss: str = self.accepted_losses[loss_fn]
        """The loss function computed between the model variables and the learner predictions."""

    def build_model(self, macs, model, x: pd.DataFrame, y: pd.Series, iteration: Iteration) -> Info:
        # if the model has not been fitted yet (i.e., the initial macs step is 'projection') we use the original labels
        # otherwise we use the predicted classes and, optionally, we include the probabilities
        pred = np.clip(macs.predict(x), a_min=self.clip_value, a_max=1 - self.clip_value) if macs.fitted else None

        # define variables and max_count (i.e., upper bound for number of counts for a class)
        classes = np.unique(y)
        num_samples = len(y)
        num_classes = len(classes)
        max_count = np.ceil(self.slack * num_samples / num_classes)
        variables = model.binary_var_matrix(keys1=num_samples, keys2=classes, name='y').values()
        variables = np.array(list(variables)).reshape(num_samples, num_classes)

        # constrain the class counts to the maximal value
        for c in range(num_classes):
            class_count = model.sum([variables[i, c] for i in range(num_samples)])
            model.add(class_count <= max_count)
        # each sample should be labeled with one class only
        for i in range(num_samples):
            class_label = model.sum(variables[i, c] for c in range(num_classes))
            model.add(class_label == 1)

        # return model info
        return BalancedCounts.Info(variables=variables, predictions=pred, max_count=max_count)

    def beta_step(self, macs, model, x: pd.DataFrame, y: pd.Series, model_info: Info, iteration: Iteration) -> bool:
        _, pred, max_count = model_info
        # if either beta is None or there are no predictions (due to initial projection step) we use alpha, otherwise
        # we compute the class counts on the output classes and return True if they are all under the maximal value
        if self.beta is None or pred is None:
            return False
        else:
            _, classes_counts = np.unique(Classifier.get_classes(pred), return_counts=True)
            return np.all(classes_counts <= max_count)

    def y_loss(self, macs, model, x: pd.DataFrame, y: pd.Series, model_info: Info, iteration: Iteration) -> Any:
        variables, _, _ = model_info
        return BalancedCounts.losses.categorical_hamming(model=model, numeric_variables=y, model_variables=variables)

    def p_loss(self, macs, model, x: pd.DataFrame, y: pd.Series, model_info: Info, iteration: Iteration) -> Any:
        model_variables, pred, _ = model_info
        if pred is None:
            return 0.0
        else:
            # if the chosen loss is hamming distance we use classes instead of probabilities
            pred = Classifier.get_classes(pred) if self._p_loss == 'categorical_hamming' else pred
            # retrieve the loss from the losses handler by name
            loss_fn = getattr(BalancedCounts.losses, self._p_loss)
            return loss_fn(model=model, numeric_variables=pred, model_variables=model_variables)

    def return_solutions(self, macs, solution, x: pd.DataFrame, y: pd.Series, model_info: Info,
                         iteration: Iteration) -> np.ndarray:
        variables, _, _ = model_info
        return np.array([[v.solution_value for v in row] for row in variables]).argmax(axis=1)
