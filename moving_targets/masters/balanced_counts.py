"""Master implementation for the Balance Counts problem."""

import numpy as np

from moving_targets.masters.cplex_master import CplexMaster
from moving_targets.utils.typing import Matrix, Vector, Iteration


class BalancedCounts(CplexMaster):
    """Cplex Master for the Balanced Counts problem in which output classes are constrained to be equally distributed.

    Args:
        n_classes: the number of output classes.
        use_prob: whether to use the learner's probabilities or the learner's output classes directly in the p_loss.
        clip_value: the clipping value for probabilities, in case they are used.
        slack: the slack value, expressed in terms of ratio wrt the other classes, to allow some output class to have a
               few labelled samples more than the other classes.
        **kwargs: super-class arguments.
    """

    def __init__(self, n_classes: int, use_prob: bool = True, clip_value: float = 1e-2, slack: float = 1.05, **kwargs):
        super(BalancedCounts, self).__init__(**kwargs)
        self.num_classes: int = n_classes
        self.use_prob: bool = use_prob
        self.clip_value: float = clip_value
        self.slack = slack

    # noinspection PyMissingOrEmptyDocstring
    def build_model(self, macs, model, x: Matrix, y: Vector, iteration: Iteration) -> object:
        # if the model has not been fitted yet (i.e., the initial macs step is 'projection') we use the original labels
        # otherwise we use either the predicted classes or the predicted probabilities
        if not macs.fitted:
            prob = None
            pred = y.reshape(-1, )
        elif self.use_prob is False:
            prob = None
            pred = macs.learner.predict(x)
        else:
            assert hasattr(macs.learner, 'predict_proba'), "Learner must have method 'predict_proba(x)' for use_prob"
            prob = np.clip(macs.learner.predict_proba(x), a_min=self.clip_value, a_max=1 - self.clip_value)
            pred = macs.learner.predict(x)

        # define variables and max_count (i.e., upper bound for number of counts for a class)
        num_samples = len(y)
        max_count = np.ceil(self.slack * num_samples / self.num_classes)
        variables = model.binary_var_matrix(keys1=num_samples, keys2=self.num_classes, name='y').values()
        variables = np.array(list(variables)).reshape(num_samples, self.num_classes)

        # constrain the class counts to the maximal value
        for c in range(self.num_classes):
            class_count = model.sum([variables[i, c] for i in range(num_samples)])
            model.add_constraint(class_count <= max_count)
        # each sample should be labeled with one class only
        for i in range(num_samples):
            class_label = model.sum(variables[i, c] for c in range(self.num_classes))
            model.add_constraint(class_label == 1)

        # return model info
        return variables, pred, prob, max_count

    # noinspection PyMissingOrEmptyDocstring
    def beta_step(self, macs, model, model_info: object, x: Matrix, y: Vector, iteration: Iteration) -> bool:
        _, pred, _, max_count = model_info
        _, pred_classes_counts = np.unique(pred, return_counts=True)
        return np.all(pred_classes_counts <= max_count)

    # noinspection PyMissingOrEmptyDocstring
    def y_loss(self, macs, model, model_info, x: Matrix, y: Vector, iteration: Iteration) -> float:
        variables, _, _, _ = model_info
        return CplexMaster.losses.categorical_hamming(model=model, numeric_variables=y, model_variables=variables)

    # noinspection PyMissingOrEmptyDocstring
    def p_loss(self, macs, model, model_info, x: Matrix, y: Vector, iteration: Iteration) -> float:
        variables, pred, prob, _ = model_info
        if prob is None:
            return CplexMaster.losses.categorical_hamming(
                model=model,
                numeric_variables=pred,
                model_variables=variables
            )
        else:
            return CplexMaster.losses.categorical_crossentropy(
                model=model,
                numeric_variables=prob,
                model_variables=variables
            )

    # noinspection PyMissingOrEmptyDocstring
    def return_solutions(self, macs, solution, model_info, x: Matrix, y: Vector, iteration: Iteration) -> object:
        variables, _, _, _ = model_info
        y_adj = [sum(c * solution.get_value(variables[i, c]) for c in range(self.num_classes)) for i in range(len(y))]
        y_adj = np.array([int(v) for v in y_adj])
        return y_adj
