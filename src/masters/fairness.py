"""Master implementation for the Fair Regression problem."""
from collections import namedtuple
from typing import Optional

import numpy as np
import pandas as pd

from moving_targets.learners.learner import Classifier
from moving_targets.masters.cplex_master import CplexMaster
from moving_targets.metrics import DIDI
from moving_targets.util.typing import Iteration, Solution


class Fairness(CplexMaster):
    """Master for the Fairness problems."""

    Info = namedtuple('Info', 'variables predictions')
    """Data structure for the model info returned by the 'build_model()' method."""

    def __init__(self, classification: bool, protected: str, y_loss: str, p_loss: str, violation: float, alpha: float,
                 beta: float, time_limit: Optional[float]):
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

        self.didi: DIDI = DIDI(classification=classification, protected=protected)
        """A `DisparateImpactDiscriminationIndex` metric object used to compute the DIDI."""

        self._y_loss: str = y_loss
        """The loss function computed between the model variables and the original targets."""

        self._p_loss: str = p_loss
        """The loss function computed between the model variables and the learner predictions."""

    def build_model(self, macs, model, x: pd.DataFrame, y: pd.Series, iteration: Iteration) -> Info:
        raise NotImplementedError("Please implement abstract method 'build_model'")

    def beta_step(self, macs, model, x: pd.DataFrame, y: pd.Series, model_info: Info, iteration: Iteration) -> bool:
        _, pred = model_info
        # if either beta is None or there are no predictions (due to initial projection step) we use alpha
        # otherwise we compute the didi using the metric, then returns True if the violation is under the threshold
        if self.beta is None or pred is None:
            return False
        else:
            return self.didi(x=x, y=y, p=pred) <= self.violation

    def y_loss(self, macs, model, x: pd.DataFrame, y: pd.Series, model_info: Info, iteration: Iteration) -> float:
        var, _ = model_info
        y_loss = getattr(CplexMaster.losses, self._y_loss)
        return y_loss(model=model, numeric_variables=y, model_variables=var)

    def p_loss(self, macs, model, x: pd.DataFrame, y: pd.Series, model_info: Info, iteration: Iteration) -> float:
        variables, pred = model_info
        if pred is None:
            return 0.0
        else:
            return CplexMaster.losses.mean_squared_error(model=model, numeric_variables=pred, model_variables=variables)

    def return_solutions(self, macs, solution, x: pd.DataFrame, y: pd.Series, model_info: Info,
                         iteration: Iteration) -> Solution:
        raise NotImplementedError("Please implement abstract method 'return_solutions'")


class FairRegression(Fairness):
    """Master for the Fairness Regression problem."""

    losses = {
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
        assert loss_fn in self.losses.keys(), f"'{loss_fn}' is not a valid loss function"
        super(FairRegression, self).__init__(classification=False, protected=protected, y_loss=self.losses[loss_fn],
                                             p_loss=self.losses[loss_fn], violation=violation, alpha=alpha, beta=beta,
                                             time_limit=time_limit)

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
        deviations = model.continuous_var_list(keys=indicator_matrix.columns, lb=0.0, name='deviations')
        for idx, label in enumerate(indicator_matrix.columns):
            # subset of the variables having <label> as protected feature (i.e., the current protected group)
            protected_variables = variables[indicator_matrix[label] == 1]
            if len(protected_variables) > 0:
                # average output target for the protected group
                protected_average = model.sum(protected_variables) / len(protected_variables)
                # average output target for the whole dataset
                total_average = model.sum(variables) / len(variables)
                # linearize with constraints the absolute value of the deviation between the total average samples
                # and the average samples within the protected group in order to consider the unsigned value
                model.add_constraint(deviations[idx] >= total_average - protected_average)
                model.add_constraint(deviations[idx] >= protected_average - total_average)

        # the DIDI is computed as the sum of this deviations, and it is constrained to be lower than the given value
        didi = model.sum(deviations)
        model.add_constraint(didi <= self.violation, ctname='fairness_constraint')

        # return model info
        return Fairness.Info(variables=variables, predictions=pred)

    def return_solutions(self, macs, solution, x: pd.DataFrame, y: pd.Series, model_info: Fairness.Info,
                         iteration: Iteration) -> Solution:
        variables, _ = model_info
        return np.array([v.solution_value for v in variables])


class FairClassification(Fairness):
    """Master for the Fairness Classification problem."""

    losses = {
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
        assert loss_fn in self.losses.keys(), f"'{loss_fn}' is not a valid loss function"
        super(FairClassification, self).__init__(classification=True, protected=protected, y_loss='categorical_hamming',
                                                 p_loss=self.losses[loss_fn], violation=violation, alpha=alpha,
                                                 beta=beta, time_limit=time_limit)

        self.clip_value: float = clip_value
        """The clipping value for probabilities, in case they are used."""

    def build_model(self, macs, model, x: pd.DataFrame, y: pd.Series, iteration: Iteration) -> Fairness.Info:
        # retrieve predictions and indicator matrix
        pred = macs.predict(x) if macs.fitted else None
        indicator_matrix = self.didi.get_indicator_matrix(x=x)

        # define model variables
        classes = np.unique(y)
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
                    # linearize with constraints the absolute value of the deviation between the total average samples
                    # and the average samples within the protected group in order to consider the unsigned value
                    model.add_constraint(deviations[idx, class_idx] >= total_average - protected_average)
                    model.add_constraint(deviations[idx, class_idx] >= protected_average - total_average)

        # the DIDI is computed as the sum of this deviations, and it is constrained to be lower than the given value
        didi = model.sum(deviations)
        model.add_constraint(didi <= self.violation, ctname='fairness_constraint')

        # return model info
        return Fairness.Info(variables=variables, predictions=pred)

    def p_loss(self, macs, model, x: pd.DataFrame, y: pd.Series, model_info: Fairness.Info,
               iteration: Iteration) -> float:
        variables, pred = model_info
        pred = Classifier.get_classes(pred) if pred is not None and self._p_loss == 'categorical_hamming' else pred
        return super(FairClassification, self).p_loss(macs=macs, model=model, x=x, y=y, iteration=iteration,
                                                      model_info=Fairness.Info(variables=variables, predictions=pred))

    def return_solutions(self, macs, solution, x: pd.DataFrame, y: pd.Series, model_info: Fairness.Info,
                         iteration: Iteration) -> Solution:
        variables, _ = model_info
        return np.array([[v.solution_value for v in row] for row in variables]).argmax(axis=1)
