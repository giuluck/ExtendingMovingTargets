"""Master implementation for the Balance Counts problem."""
from typing import Any, Optional

import numpy as np

from moving_targets.learners.learner import Classifier
from moving_targets.masters.cplex_master import CplexMaster
from moving_targets.util.typing import Matrix, Vector, Iteration


class BalancedCounts(CplexMaster):
    """Master for the Balanced Counts problem in which output classes are constrained to be equally distributed."""

    losses = {
        'hd': 'categorical_hamming',
        'hamming_distance': 'categorical_hamming',
        'categorical_hamming': 'categorical_hamming',
        'ce': 'categorical_crossentropy',
        'cce': 'categorical_crossentropy',
        'crossentropy': 'categorical_crossentropy',
        'categorical_crossentropy': 'categorical_crossentropy',
        'mae': 'mean_absolute_error',
        'mean_absolute_error': 'mean_absolute_error',
        'mse': 'mean_squared_error',
        'mean_squared_error': 'mean_squared_error'
    }
    """A dictionary of losses aliases."""

    def __init__(self, num_classes: Optional[int] = None, loss_fn: str = 'hd', clip_value: float = 1e-2,
                 slack: float = 1.05, alpha: float = 1., beta: float = 1., time_limit: Optional[float] = None):
        """
        :param alpha:
            The non-negative real number which is used to calibrate the two losses in the alpha step.

        :param beta:
            The non-negative real number which is used to constraint the p_loss in the beta step.

        :param time_limit:
            The maximal time for which the master can run during each iteration.

        :param num_classes:
            The number of output classes. If None, infers it from the input data.

        :param loss_fn:
            The loss function computed between the model variables and the learner predictions.

            Must be one in ['hamming_distance', 'crossentropy', 'mean_squared_error' and 'mean_absolute_error'].

        :param clip_value:
            The clipping value for probabilities, in case they are used.

        :param slack:
            The slack value, expressed in terms of ratio wrt the other classes, to allow some output class to have a
            few labelled samples more than the other classes.
        """
        assert loss_fn in self.losses.keys(), f"'{loss_fn}' is not a valid loss function"
        super(BalancedCounts, self).__init__(alpha=alpha, beta=beta, time_limit=time_limit)

        self.num_classes: Optional[int] = num_classes
        """The number of output classes."""

        self.loss_fn: str = self.losses[loss_fn]
        """The loss function computed between the model variables and the learner predictions."""

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

        :return:
            A tuple containing:

            1. the list of model variables;
            2. the vector of learner predictions;
            3. the vector of learner probabilities;
            4. the maximal number of elements for each class in order to achieve balance.
        """
        # if the model has not been fitted yet (i.e., the initial macs step is 'projection') we use the original labels
        # otherwise we use the predicted classes and, optionally, we include the probabilities
        if not macs.fitted:
            prob = None
            pred = y
        else:
            prob = np.clip(macs.learner.predict(x), a_min=self.clip_value, a_max=1 - self.clip_value)
            pred = Classifier.get_classes(prob)

        # define variables and max_count (i.e., upper bound for number of counts for a class)
        num_samples = len(y)
        num_classes = len(np.unique(y)) if self.num_classes is None else self.num_classes
        max_count = np.ceil(self.slack * num_samples / num_classes)
        variables = model.binary_var_matrix(keys1=num_samples, keys2=num_classes, name='y').values()
        variables = np.array(list(variables)).reshape(num_samples, num_classes)

        # constrain the class counts to the maximal value
        for c in range(num_classes):
            class_count = model.sum([variables[i, c] for i in range(num_samples)])
            model.add_constraint(class_count <= max_count)
        # each sample should be labeled with one class only
        for i in range(num_samples):
            class_label = model.sum(variables[i, c] for c in range(num_classes))
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

        :return:
            A boolean value that decides whether or not to use the beta step during the current iteration.
        """
        if self._beta is None:
            return False
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

        :return:
            A real number representing the categorical hamming distance/crossentropy loss.
        """
        model_variables, pred, prob, _ = model_info
        # if no probabilities are available (i.e., macs is not fitted yet due to pretraining) or the chosen loss is
        # hamming distance (which uses class labels instead of probabilities), we use hamming distance on predictions
        if prob is None or self.loss_fn == 'categorical_hamming':
            return CplexMaster.losses.categorical_hamming(
                model=model,
                numeric_variables=pred,
                model_variables=model_variables
            )
        else:
            # retrieve the loss from the losses handler by name
            loss_fn = getattr(CplexMaster.losses, self.loss_fn)
            return loss_fn(
                model=model,
                numeric_variables=prob,
                model_variables=model_variables
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

        :return:
            The vector of adjusted targets returned by the optimization model.
        """
        variables, _, _, _ = model_info
        num_samples, num_classes = variables.shape
        y_adj = [sum(c * solution.get_value(variables[i, c]) for c in range(num_classes)) for i in range(num_samples)]
        y_adj = np.array([int(v) for v in y_adj])
        return y_adj
