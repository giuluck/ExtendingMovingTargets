"""Basic Master Interface."""
from typing import Any, Optional

from moving_targets.util.typing import Iteration, Solution


class Master:
    """Basic interface for a Moving Targets learner."""

    def __init__(self, alpha: Optional[float] = 1., beta: Optional[float] = 1.):
        """
        :param alpha:
            The initial positive real number which is used to calibrate the two losses in the alpha step.
        
        :param beta:
            The initial non-negative real number which is used to constraint the p_loss in the beta step.
        """
        super(Master, self).__init__()

        self._alpha: Optional[float] = alpha
        """The initial positive real number which is used to calibrate the two losses in the alpha step."""

        self._beta: Optional[float] = beta
        """The initial non-negative real number which is used to constraint the p_loss in the beta step."""

    def build_model(self, macs, model, x, y, iteration: Iteration) -> Any:
        """Creates the model variables and adds the problem constraints.

        :param macs:
            Reference to the `MACS` object encapsulating the master.
        
        :param model:
            The inner optimization model.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param iteration:
            The current `MACS` iteration, usually a number.

        :return:
            Any object containing information that may be useful in other methods.
        """
        raise NotImplementedError("Please implement abstract method 'build_model'")

    def alpha(self, macs, model, x, y, model_info, iteration: Iteration) -> float:
        """Computes the alpha for the given iteration.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param model:
            The inner optimization model.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param model_info:
            The information returned by the 'build_model' function.

        :param iteration:
            The current `MACS` iteration, usually a number.

        :return:
            Whether to use or not the beta step during the current iteration.
        """
        return 1.0 if self._alpha is None else self._alpha

    def beta(self, macs, model, x, y, model_info, iteration: Iteration) -> Optional[float]:
        """Computes the beta for the given iteration. If None is returned, the alpha step is used instead.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param model:
            The inner optimization model.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param model_info:
            The information returned by the 'build_model' function.

        :param iteration:
            The current `MACS` iteration, usually a number.

        :return:
            Whether to use or not the beta step during the current iteration.
        """
        return self._beta

    def y_loss(self, macs, model, x, y, model_info, iteration: Iteration) -> Any:
        """Computes the loss of the model variables wrt real targets.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param model:
            The inner optimization model.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param model_info:
            The information returned by the 'build_model' function.

        :param iteration:
            The current `MACS` iteration, usually a number.

        :return:
            A real number representing the y_loss.
        """
        return 0.0

    def p_loss(self, macs, model, x, y, model_info, iteration: Iteration) -> Any:
        """Computes the loss of the model variables wrt predictions.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param model:
            The inner optimization model.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param model_info:
            The information returned by the 'build_model' function.

        :param iteration:
            The current `MACS` iteration, usually a number.

        :return:
            A real number representing the p_loss.
        """
        return 0.0

    def return_solutions(self, macs, solution, x, y, model_info, iteration: Iteration) -> Solution:
        """Processes and returns the solutions given by the optimization model.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param solution:
            The object containing information about the solution of the problem.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param model_info:
            The information returned by the 'build_model' function.

        :param iteration:
            The current `MACS` iteration, usually a number.

        :return:
            Either a simple vector of adjusted targets or a tuple containing the vector and a dictionary of kwargs.
        """
        raise NotImplementedError("Please implement abstract method 'return_solutions'")

    def adjust_targets(self, macs, x, y, iteration: Iteration) -> Solution:
        """Core function of the Master object. Builds the model dependently on the kind solver and returns the adjusted
        targets.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param iteration:
            The current `MACS` iteration, usually a number.

        :return:
            The vector of adjusted targets, potentially with a dictionary of additional information.
        """
        raise NotImplementedError("Please implement abstract method 'adjust_targets'")
