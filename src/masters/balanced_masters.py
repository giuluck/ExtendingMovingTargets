"""Master implementation for the Balance Counts problem."""

import numpy as np
from moving_targets.masters import SingleTargetClassification
from moving_targets.masters.backends import Backend
from moving_targets.util import probabilities


class BalancedCounts(SingleTargetClassification):
    """Master for the Balanced Counts problem in which output classes are constrained to be equally distributed."""

    def __init__(self, backend: Backend, loss: str, alpha: float, beta: float, adaptive: bool):
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

        # noinspection PyUnusedLocal
        def satisfied(x, y, p):
            # constraint is satisfied if all the classes counts are lower than then average number of counts per class
            pred = probabilities.get_classes(p)
            classes, counts = np.unique(pred, return_counts=True)
            max_count = np.ceil(len(y) / len(classes))
            return np.all(counts <= max_count)

        super().__init__(backend=backend, satisfied=satisfied, alpha=alpha, beta=beta,
                         y_loss='hd', p_loss=loss, stats=True)

        self.adaptive: bool = adaptive
        """Whether or not to use an adaptive strategy for the alpha value."""

    def build(self, x, y, p):
        # retrieve the variables
        variables = super(BalancedCounts, self).build(x, y, p)
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

    def alpha(self, x, y, p, v) -> float:
        alpha = super(BalancedCounts, self).alpha(x, y, p, v)
        if self.adaptive:
            return alpha / self._macs.iteration
        else:
            return alpha
