"""Master implementations for the various tasks."""
from typing import Optional

import numpy as np
from moving_targets.masters import RegressionMaster, ClassificationMaster
from moving_targets.masters.backends import Backend
from moving_targets.masters.optimizers import ConstantSlope
from moving_targets.metrics import DIDI
from moving_targets.util import probabilities


class BalancedCounts(ClassificationMaster):
    """Master for the Balanced Counts problem in which output classes are constrained to be equally distributed."""

    def __init__(self, backend: Backend, loss: str, alpha: float, beta: Optional[float], adaptive: bool):
        """
        :param backend:
            The backend instance or backend alias.

        :param loss:
            The loss function computed between the model variables and the learner predictions.

            Must be one in ['hamming_distance', 'crossentropy', 'mean_squared_error' and 'mean_absolute_error'].

        :param alpha:
            The non-negative real number which is used to calibrate the two losses in the alpha step.

        :param beta:
            The non-negative real number which is used to constraint the p_loss in the beta step.

        :param adaptive:
            Whether or not to use an adaptive strategy for the alpha value.
        """
        alpha = ConstantSlope(base=alpha) if adaptive else alpha
        super().__init__(backend=backend, y_loss='hd', p_loss=loss, alpha=alpha, beta=beta, stats=True)

    def use_beta(self, x, y: np.ndarray, p: np.ndarray) -> bool:
        # constraint is satisfied if all the classes counts are lower than then average number of counts per class
        pred = probabilities.get_discrete(p, task='classification')
        classes, counts = np.unique(pred, return_counts=True)
        max_count = np.ceil(len(y) / len(classes))
        return np.all(counts <= max_count)

    def build(self, x, y: np.ndarray) -> np.ndarray:
        # retrieve the variables
        variables = super(BalancedCounts, self).build(x, y)
        # compute the upper bound for number of counts of a class, which will be used to constraint the model variables
        classes = np.unique(y)
        max_count = np.ceil(len(y) / len(classes))
        # constraint the model variables by computing the sum of each column
        # (the np.atleast_2d() call is used to handle binary classification tasks, where variables is a 1d vector)
        constraints = []
        for column in np.atleast_2d(variables).transpose():
            class_count = self.backend.sum(column)
            constraints.append(class_count <= max_count)
        self.backend.add_constraints(constraints=constraints)
        # return the variables
        return variables


class FairClassification(ClassificationMaster):
    """Master for the Fairness Classification problem."""

    def __init__(self, protected: str, backend: Backend, loss: str, alpha: float, beta: float, adaptive: bool):
        """
        :param protected:
            The name of the protected feature.

        :param backend:
            The backend instance or backend alias.

        :param loss:
            The loss function used in the master problem for both the y_loss and the p_loss.

            Must be either 'mean_squared_error' or 'mean_absolute_error'.

        :param alpha:
            The non-negative real number which is used to calibrate the two losses in the alpha step.

        :param beta:
            The non-negative real number which is used to constraint the p_loss in the beta step.

        :param adaptive:
            Whether or not to use an adaptive strategy for the alpha value.
        """

        self.violation = 0.2
        """The maximal accepted level of violation of the constraint."""

        self.didi = DIDI(protected=protected, classification=True, percentage=True)
        """A DIDI metric instance used to compute both the indicator matrices and the satisfiability."""

        alpha = ConstantSlope(base=alpha) if adaptive else alpha
        super().__init__(backend=backend, y_loss='hd', p_loss=loss, alpha=alpha, beta=beta, stats=True)

    def use_beta(self, x, y: np.ndarray, p: np.ndarray) -> bool:
        # the constraint is satisfied if the percentage DIDI is lower or equal to the expected violation
        return self.didi(x=x, y=y, p=p) <= self.violation

    def build(self, x, y: np.ndarray) -> np.ndarray:
        # retrieve model variables from the super method and, for compatibility between the binary/multiclass scenarios,
        # optionally transform 1d variables (representing a binary classification task) into a 2d matrix
        super_vars = super(FairClassification, self).build(x, y)
        variables = np.transpose([1 - super_vars, super_vars]) if super_vars.ndim == 1 else super_vars

        # as a first step, we need to compute the deviations between the average output for the total dataset and the
        # average output respectively to each protected class
        indicator_matrix = DIDI.get_indicator_matrix(x=x, protected=self.didi.protected)
        groups, classes = len(indicator_matrix), len(np.unique(y))
        deviations = self.backend.add_continuous_variables(groups, classes, lb=0.0, name='deviations')
        # this is the average number of samples from the whole dataset <class[c]> as target class
        total_avg = [self.backend.sum(variables[:, c]) / len(variables) for c in range(classes)]
        for g, protected_group in enumerate(indicator_matrix):
            # this is the subset of the variables having <label> as protected feature (i.e., the protected group)
            protected_vars = variables[protected_group]
            if len(protected_vars) == 0:
                continue
            # this is the average number of samples within the protected group having <class[c]> as target
            protected_avg = [self.backend.sum(protected_vars[:, c]) / len(protected_vars) for c in range(classes)]
            # eventually, the partial deviation is computed as the absolute value (which is linearized) of the
            # difference between the total average samples and the average samples within the protected group
            self.backend.add_constraints([deviations[g, c] >= total_avg[c] - protected_avg[c] for c in range(classes)])
            self.backend.add_constraints([deviations[g, c] >= protected_avg[c] - total_avg[c] for c in range(classes)])
        # finally, we compute the DIDI as the sum of this deviations, which is constrained to be lower or equal to the
        # given value (also, since we are computing the percentage DIDI, we need to scale for the original train_didi)
        didi = self.backend.sum(deviations)
        train_didi = DIDI.classification_didi(indicator_matrix=indicator_matrix, targets=y)
        self.backend.add_constraint(didi <= self.violation * train_didi)
        return super_vars


class FairRegression(RegressionMaster):
    """Master for the Fairness Regression problem."""

    def __init__(self,
                 protected: str,
                 backend: Backend,
                 loss: str,
                 alpha: float,
                 beta: float,
                 lb: float,
                 ub: float,
                 adaptive: bool):
        """
        :param protected:
            The name of the protected feature.

        :param backend:
            The backend instance or backend alias.

        :param loss:
            The loss function used in the master problem for both the y_loss and the p_loss.

            Must be either 'mean_squared_error' or 'mean_absolute_error'.

        :param alpha:
            The non-negative real number which is used to calibrate the two losses in the alpha step.

        :param beta:
            The non-negative real number which is used to constraint the p_loss in the beta step.

        :param lb:
            The model variables lower bounds.

        :param ub:
            The model variables upper bounds.

        :param adaptive:
            Whether or not to use an adaptive strategy for the alpha value.
        """

        self.violation = 0.2
        """The maximal accepted level of violation of the constraint."""

        self.didi = DIDI(protected=protected, classification=False, percentage=True)
        """A DIDI metric instance used to compute both the indicator matrices and the satisfiability."""

        alpha = ConstantSlope(base=alpha) if adaptive else alpha
        super().__init__(backend=backend, y_loss=loss, p_loss=loss, alpha=alpha, beta=beta, lb=lb, ub=ub, stats=True)

    def use_beta(self, x, y: np.ndarray, p: np.ndarray) -> bool:
        # the constraint is satisfied if the percentage DIDI is lower or equal to the expected violation; moreover,
        # since we know that the predictions must be positive, so we clip them to 0.0 in order to avoid (wrong)
        # negative predictions to influence the satisfiability computation
        return self.didi(x=x, y=y, p=p.clip(0.0)) <= self.violation

    def build(self, x, y: np.ndarray) -> np.ndarray:
        variables = super(FairRegression, self).build(x, y)

        # as a first step, we need to compute the deviations between the average output for the total dataset and the
        # average output respectively to each protected class
        indicator_matrix = DIDI.get_indicator_matrix(x=x, protected=self.didi.protected)
        deviations = self.backend.add_continuous_variables(len(indicator_matrix), lb=0.0, name='deviations')
        # this is the average output target for the whole dataset
        total_avg = self.backend.sum(variables) / len(variables)
        for g, protected_group in enumerate(indicator_matrix):
            # this is the subset of the variables having <label> as protected feature (i.e., the protected group)
            protected_vars = variables[protected_group]
            if len(protected_vars) == 0:
                continue
            # this is the average output target for the protected group
            protected_avg = self.backend.sum(protected_vars) / len(protected_vars)
            # eventually, the partial deviation is computed as the absolute value (which is linearized) of the
            # difference between the total average samples and the average samples within the protected group
            self.backend.add_constraint(deviations[g] >= total_avg - protected_avg)
            self.backend.add_constraint(deviations[g] >= protected_avg - total_avg)

        # finally, we compute the DIDI as the sum of this deviations, which is constrained to be lower or equal to the
        # given value (also, since we are computing the percentage DIDI, we need to scale for the original train_didi)
        didi = self.backend.sum(deviations)
        train_didi = DIDI.regression_didi(indicator_matrix=indicator_matrix, targets=y)
        self.backend.add_constraint(didi <= self.violation * train_didi)
        return variables
