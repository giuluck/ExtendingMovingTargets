"""Master implementation for the Fair Regression problem."""
from collections import namedtuple
from typing import Optional, Dict, Any, Callable

import numpy as np
import pandas as pd

from moving_targets.learners.learner import Classifier
from moving_targets.masters import LossesHandler
from moving_targets.masters.cplex_master import CplexMaster
from moving_targets.metrics import DIDI
from moving_targets.util.typing import Iteration, Solution


class Fairness(CplexMaster):
    """Master for the Fairness problems."""

    Info = namedtuple('Info', 'variables predictions')
    """Data structure for the model info returned by the 'build_model()' method."""

    def __init__(self, classification: bool, protected: str, y_loss: Callable, p_loss: Callable, violation: float,
                 alpha: float, beta: float, time_limit: Optional[float]):
        """
        :param classification:
            Whether the task is a classification (True) or a regression (False) task.

        :param protected:
            The name of the protected feature.

        :param y_loss:
            The loss function computed between the model variables and the original targets.

        :param p_loss:
            The loss function computed between the model variables and the learner predictions.

        :param violation:
            The maximal accepted level of violation of the constraint.

        :param alpha:
            The non-negative real number which is used to calibrate the two losses in the alpha step.

        :param beta:
            The non-negative real number which is used to constraint the p_loss in the beta step.

        :param time_limit:
            The maximal time for which the master can run during each iteration.
        """
        super(Fairness, self).__init__(alpha=alpha, beta=beta, time_limit=time_limit)

        self.violation: float = violation
        """The maximal accepted level of violation of the constraint."""

        self.didi: DIDI = DIDI(classification=classification, protected=protected, percentage=True)
        """A `DisparateImpactDiscriminationIndex` metric object used to compute the DIDI."""

        self._y_loss: Callable = y_loss
        """The loss function computed between the model variables and the original targets."""

        self._p_loss: Callable = p_loss
        """The loss function computed between the model variables and the learner predictions."""

    def build_model(self, macs, model, x: pd.DataFrame, y: pd.Series, iteration: Iteration) -> Info:
        raise NotImplementedError("Please implement abstract method 'build_model'")

    def beta_step(self, macs, model, x: pd.DataFrame, y: pd.Series, model_info: Info, iteration: Iteration) -> bool:
        _, pred = model_info
        # if either beta is None or there are no predictions (due to initial projection step) we use alpha
        # otherwise we compute the didi using the metric, then returns True if the violation is under the threshold
        return False if self.beta is None or pred is None else self.didi(x=x, y=y, p=pred) <= self.violation

    def y_loss(self, macs, model, x: pd.DataFrame, y: pd.Series, model_info: Info, iteration: Iteration) -> Any:
        var, _ = model_info
        return self._y_loss(model=model, numeric_variables=y, model_variables=var)

    def p_loss(self, macs, model, x: pd.DataFrame, y: pd.Series, model_info: Info, iteration: Iteration) -> Any:
        variables, pred = model_info
        return 0.0 if pred is None else self._p_loss(model=model, numeric_variables=pred, model_variables=variables)

    def return_solutions(self, macs, solution, x: pd.DataFrame, y: pd.Series, model_info: Info,
                         iteration: Iteration) -> Solution:
        raise NotImplementedError("Please implement abstract method 'return_solutions'")


class FairRegression(Fairness):
    """Master for the Fairness Regression problem."""

    accepted_losses: Dict[str, str] = {
        'mae': 'mean_absolute_error',
        'mean_absolute_error': 'mean_absolute_error',
        'mse': 'mean_squared_error',
        'mean_squared_error': 'mean_squared_error'
    }
    """A dictionary of losses aliases that are accepted."""

    def __init__(self, protected: str, loss_fn: str = 'mse', violation: float = 0.2, lb: float = 0.0,
                 ub: float = float('inf'), alpha: float = 1., beta: float = 1., time_limit: Optional[float] = 30):
        """
        :param protected:
            The name of the protected feature.

        :param loss_fn:
            The loss function used in the master problem for both the y_loss and the p_loss.

            Must be either 'mean_squared_error' or 'mean_absolute_error'.

        :param violation:
            The maximal accepted level of violation of the constraint.

        :param lb:
            The variables' lower bound.

        :param ub:
            The variables' upper bound.

        :param alpha:
            The non-negative real number which is used to calibrate the two losses in the alpha step.

        :param beta:
            The non-negative real number which is used to constraint the p_loss in the beta step.

        :param time_limit:
            The maximal time for which the master can run during each iteration.
        """
        assert loss_fn in self.accepted_losses.keys(), f"'{loss_fn}' is not a valid loss function"
        _loss = getattr(FairRegression.losses, FairRegression.accepted_losses[loss_fn])
        super(FairRegression, self).__init__(classification=False, protected=protected, violation=violation,
                                             y_loss=_loss, p_loss=_loss, alpha=alpha, beta=beta, time_limit=time_limit)

        self.lb: float = lb
        """The variables' lower bound."""

        self.ub: float = ub
        """The variables' upper bound."""

    def build_model(self, macs, model, x: pd.DataFrame, y: pd.Series, iteration: Iteration) -> Fairness.Info:
        # retrieve predictions and indicator matrix, then define model variables
        pred = macs.predict(x) if macs.fitted else None
        variables = np.array(model.continuous_var_list(keys=len(y), lb=self.lb, ub=self.ub, name='y'))
        indicator_matrix = self.didi.get_indicator_matrix(x=x)

        # calculate deviations between the average output and the average output respectively to each protected class
        deviations = model.continuous_var_list(keys=indicator_matrix.columns, name='deviations')
        for idx, label in enumerate(indicator_matrix.columns):
            # subset of the variables having <label> as protected feature (i.e., the current protected group)
            protected_variables = variables[indicator_matrix[label] == 1]
            if len(protected_variables) > 0:
                # average output target for the protected group
                protected_average = model.sum(protected_variables) / len(protected_variables)
                # average output target for the whole dataset
                total_average = model.sum(variables) / len(variables)
                # the partial deviation is computed as the absolute value of the difference between the
                # total average samples and the average samples within the protected group
                deviations[idx] = model.abs(total_average - protected_average)

        # the DIDI is computed as the sum of this deviations, and it is constrained to be lower than the given value
        train_didi = DIDI.regression_didi(indicator_matrix=indicator_matrix, targets=y)
        didi = model.sum(deviations) / train_didi
        model.add_constraint(didi <= self.violation, ctname='fairness_constraint')

        # return model info
        return Fairness.Info(variables=variables, predictions=pred)

    def return_solutions(self, macs, solution, x: pd.DataFrame, y: pd.Series, model_info: Fairness.Info,
                         iteration: Iteration) -> Solution:
        variables, _ = model_info
        return np.array([v.solution_value for v in variables])


class FairClassification(Fairness):
    """Master for the Fairness Classification problem."""

    losses: LossesHandler = LossesHandler(sqr_fn=lambda m, mvs, nvs: (nvs * (1 - mvs) + (1 - nvs) * mvs) ** 2,
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

    def __init__(self, protected: str, loss_fn: str = 'hd', violation: float = 0.2, clip_value: float = 1e-2,
                 alpha: float = 1., beta: float = 1., time_limit: Optional[float] = 30):
        """
        :param protected:
            The name of the protected feature.

        :param loss_fn:
            The loss function computed between the model variables and the learner predictions.

            Must be one in ['hamming_distance', 'crossentropy', 'mean_squared_error' and 'mean_absolute_error'].

        :param clip_value:
            The clipping value for predicted probabilities.

        :param violation:
            The maximal accepted level of violation of the constraint.

        :param alpha:
            The non-negative real number which is used to calibrate the two losses in the alpha step.

        :param beta:
            The non-negative real number which is used to constraint the p_loss in the beta step.

        :param time_limit:
            The maximal time for which the master can run during each iteration.
        """
        assert loss_fn in self.accepted_losses.keys(), f"'{loss_fn}' is not a valid loss function"
        y_loss = FairClassification.losses.categorical_hamming
        p_loss = getattr(FairClassification.losses, FairClassification.accepted_losses[loss_fn])
        super(FairClassification, self).__init__(classification=True, protected=protected, y_loss=y_loss, p_loss=p_loss,
                                                 violation=violation, alpha=alpha, beta=beta, time_limit=time_limit)

        self.clip_value: float = clip_value
        """The clipping value for probabilities, in case they are used."""

        self.use_prob: bool = FairClassification.accepted_losses[loss_fn] != 'categorical_hamming'
        """Whether to use output probabilities or output classes in the p_loss."""

    def build_model(self, macs, model, x: pd.DataFrame, y: pd.Series, iteration: Iteration) -> Fairness.Info:
        # retrieve predictions and indicator matrix
        pred = macs.predict(x) if macs.fitted else None
        indicator_matrix = self.didi.get_indicator_matrix(x=x)

        # define model variables
        classes = np.unique(y)

        print(classes)

        num_samples = len(y)
        num_classes = len(classes)
        variables = model.binary_var_matrix(keys1=num_samples, keys2=classes, name='y').values()
        variables = np.array(list(variables)).reshape(num_samples, num_classes)

        # each sample should be labeled with one class only
        for i in range(num_samples):
            class_label = model.sum(variables[i, c] for c in range(num_classes))
            model.add_constraint(class_label == 1)

        # calculate deviations between the average output and the average output respectively to each protected class
        labels = indicator_matrix.columns
        deviations = model.continuous_var_matrix(keys1=labels, keys2=classes, lb=0.0, name='deviations').values()
        deviations = np.array(list(deviations)).reshape(len(labels), num_classes)
        for idx, label in enumerate(indicator_matrix.columns):
            # subset of the variables having <label> as protected feature (i.e., the current protected group)
            protected_variables = variables[indicator_matrix[label] == 1]
            if len(protected_variables) > 0:
                for class_idx in range(num_classes):
                    # subset of variables having <class[class_idx]> as target class
                    variables_per_class = variables[:, class_idx]
                    # subset of variables having <class[class_idx]> as target class and <label> as protected feature
                    protected_variables_per_class = protected_variables[:, class_idx]
                    # average number of samples within the protected group having <class[class_idx]> as target class
                    protected_average = model.sum(protected_variables_per_class) / len(protected_variables_per_class)
                    # average number of samples from the whole dataset <class[class_idx]> as target class
                    total_average = model.sum(variables_per_class) / len(variables_per_class)
                    # the partial deviation is computed as the absolute value of the difference between the
                    # total average samples and the average samples within the protected group
                    deviations[idx, class_idx] = model.abs(total_average - protected_average)

        # the DIDI is computed as the sum of this deviations, and it is constrained to be lower than the given value
        train_didi = DIDI.classification_didi(indicator_matrix=indicator_matrix, targets=y, classes=classes)
        didi = model.sum(deviations) / train_didi
        model.add_constraint(didi <= self.violation, ctname='fairness_constraint')

        # return model info
        return Fairness.Info(variables=variables, predictions=pred)

    def y_loss(self, macs, model, x: pd.DataFrame, y: pd.Series, model_info: Fairness.Info,
               iteration: Iteration) -> Any:
        variables, _ = model_info
        return model.sum([1 - variables[i, int(c)] for i, c in enumerate(y)]) / len(y)

    def p_loss(self, macs, model, x: pd.DataFrame, y: pd.Series, model_info: Fairness.Info,
               iteration: Iteration) -> float:
        variables, pred = model_info
        if pred is None:
            loss = 0.0
        elif self.use_prob:
            pred = np.clip(pred, a_min=.01, a_max=.99)
            loss = model.sum(variables * np.log(pred))
        else:
            pred = Classifier.get_classes(pred)
            loss = model.sum([1 - variables[i, c] for i, c in enumerate(pred)])
        return loss / len(y)

    def return_solutions(self, macs, solution, x: pd.DataFrame, y: pd.Series, model_info: Fairness.Info,
                         iteration: Iteration) -> Solution:
        variables, _ = model_info
        return np.array([[v.solution_value for v in row] for row in variables]).argmax(axis=1)
