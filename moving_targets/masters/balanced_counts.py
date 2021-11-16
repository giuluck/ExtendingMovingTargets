"""Master implementation for the Balance Counts problem."""
from typing import Any, Optional

import numpy as np

from moving_targets.masters.cplex_master import CplexMaster
from moving_targets.util.typing import Matrix, Vector, Iteration


class BalancedCounts(CplexMaster):
    """Master for the Balanced Counts problem in which output classes are constrained to be equally distributed."""

    def __init__(self, n_classes: int, use_prob: bool = True, clip_value: float = 1e-2, slack: float = 1.05,
                 alpha: float = 1., beta: float = 1., time_limit: Optional[float] = None):
        """
        :param alpha:
            The non-negative real number which is used to calibrate the two losses in the alpha step.

        :param beta:
            The non-negative real number which is used to constraint the p_loss in the beta step.

        :param time_limit:
            The maximal time for which the master can run during each iteration.

        :param n_classes:
            The number of output classes.

        :param use_prob:
            Whether to use the learner's probabilities or the learner's output classes directly in the p_loss.

        :param clip_value:
            The clipping value for probabilities, in case they are used.

        :param slack:
            The slack value, expressed in terms of ratio wrt the other classes, to allow some output class to have a
            few labelled samples more than the other classes.
        """
        super(BalancedCounts, self).__init__(alpha=alpha, beta=beta, time_limit=time_limit)

        self.num_classes: int = n_classes
        """The number of output classes."""

        self.use_prob: bool = use_prob
        """Whether to use the learner's probabilities or the learner's output classes directly in the p_loss."""

        self.clip_value: float = clip_value
        """The clipping value for probabilities, in case they are used."""

        self.slack: float = slack
        """The slack value, expressed in terms of ratio wrt the other classes, to allow some output class to have a
        few labelled samples more than the other classes."""

    def build_model(self, macs, model, x: Matrix, y: Vector, iteration: Iteration) -> Any:
        """Creates the model variables depending on whether or not the learner was already fitted and whether or not to
        use class probabilities to compute the loss. Then, it adds the constraints based on these variables.

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

        :returns:
            A tuple containing:

            1. the list of model variables;
            2. the vector of learner predictions;
            3. the vector of learner probabilities;
            4. the maximal number of elements for each class in order to achieve balance.
        """
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

    def beta_step(self, macs, model, model_info: object, x: Matrix, y: Vector, iteration: Iteration) -> bool:
        """Uses the model predictions and the expected maximal number of labels for each class to check if the
        balancing constraint is already satisfied; if so, returns True (beta step), otherwise returns False.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param model:
            The inner optimization model.

        :param model_info:
            The tuple returned by the 'build_model' function.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param iteration:
            The current `MACS` iteration, usually a number.

        :returns:
            A boolean value that decides whether or not to use the beta step during the current iteration.
        """
        _, pred, _, max_count = model_info
        _, pred_classes_counts = np.unique(pred, return_counts=True)
        return np.all(pred_classes_counts <= max_count)

    def y_loss(self, macs, model, model_info, x: Matrix, y: Vector, iteration: Iteration) -> Any:
        """Computes the categorical hamming distance between the model variables and the real targets (y).

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param model:
            The inner optimization model.

        :param model_info:
            The tuple returned by the 'build_model' function.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param iteration:
            The current `MACS` iteration, usually a number.

        returns:
            A real number representing the categorical hamming distance.
        """
        variables, _, _, _ = model_info
        return CplexMaster.losses.categorical_hamming(model=model, numeric_variables=y, model_variables=variables)

    def p_loss(self, macs, model, model_info, x: Matrix, y: Vector, iteration: Iteration) -> Any:
        """Computes either the categorical hamming distance or the categorical crossentropy loss (depending on whether
        or not to consider output probabilities) between the model variables and the learner's predictions.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param model:
            The inner optimization model.

        :param model_info:
            The tuple returned by the 'build_model' function.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param iteration:
            The current `MACS` iteration, usually a number.

        :returns:
            A real number representing the categorical hamming distance/crossentropy loss.
        """
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

    def return_solutions(self, macs, solution, model_info, x: Matrix, y: Vector, iteration: Iteration) -> Any:
        """Builds a numpy array from the solutions obtained from the Cplex model, the returns it.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param solution:
            The object containing information about the solution of the problem.

        :param model_info:
            The information returned by the 'build_model' function.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param iteration:
            The current `MACS` iteration, usually a number.

        :returns:
            The vector of adjusted targets returned by the optimization model.
        """
        variables, _, _, _ = model_info
        y_adj = [sum(c * solution.get_value(variables[i, c]) for c in range(self.num_classes)) for i in range(len(y))]
        y_adj = np.array([int(v) for v in y_adj])
        return y_adj
