"""Master implementation for the Fair Regression problem."""

import numpy as np
from moving_targets.masters import RegressionMaster, ClassificationMaster
from moving_targets.metrics import DIDI


class FairRegression(RegressionMaster):
    """Master for the Fairness Regression problem."""

    def __init__(self, protected, backend='gurobi', loss='mse', violation=0.2, lb=0.0, ub=None, alpha=1.0, beta=1.0):
        """
        :param protected:
            The name of the protected feature.

        :param backend:
            The backend instance or backend alias.

        :param loss:
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
        """

        self.violation = violation
        """The maximal accepted level of violation of the constraint."""

        self.didi = DIDI(protected=protected, classification=False, percentage=True)
        """A DIDI metric instance used to compute both the indicator matrices and the satisfiability."""

        # the constraint is satisfied if the percentage DIDI is lower or equal to the expected violation; moreover,
        # since we know that the predictions must be positive, so we clip them to 0.0 in order to avoid (wrong)
        # negative predictions to influence the satisfiability computation
        super().__init__(satisfied=lambda x, y, p: self.didi(x=x, y=y, p=p.clip(0.0)) <= self.violation,
                         backend=backend, lb=lb, ub=ub, alpha=alpha, beta=beta, y_loss=loss, p_loss=loss, stats=True)

    def build(self, x, y, p):
        variables = super(FairRegression, self).build(x, y, p)

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


class FairClassification(ClassificationMaster):
    """Master for the Fairness Classification problem."""

    def __init__(self, protected, backend='gurobi', loss='mse', violation=0.2, alpha=1.0, beta=1.0):
        """
        :param protected:
            The name of the protected feature.

        :param backend:
            The backend instance or backend alias.

        :param loss:
            The loss function used in the master problem for both the y_loss and the p_loss.

            Must be either 'mean_squared_error' or 'mean_absolute_error'.

        :param violation:
            The maximal accepted level of violation of the constraint.

        :param alpha:
            The non-negative real number which is used to calibrate the two losses in the alpha step.

        :param beta:
            The non-negative real number which is used to constraint the p_loss in the beta step.
        """

        self.violation = violation
        """The maximal accepted level of violation of the constraint."""

        self.didi = DIDI(protected=protected, classification=True, percentage=True)
        """A DIDI metric instance used to compute both the indicator matrices and the satisfiability."""

        # the constraint is satisfied if the percentage DIDI is lower or equal to the expected violation
        super().__init__(satisfied=lambda x, y, p: self.didi(x=x, y=y, p=p) <= self.violation,
                         backend=backend, alpha=alpha, beta=beta, y_loss='hd', p_loss=loss, stats=True)

    def build(self, x, y, p):
        # retrieve model variables from the super method and, for compatibility between the binary/multiclass scenarios,
        # optionally transform 1d variables (representing a binary classification task) into a 2d matrix
        super_vars = super(FairClassification, self).build(x, y, p)
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
