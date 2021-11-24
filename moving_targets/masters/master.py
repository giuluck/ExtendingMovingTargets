"""Basic Master Interface."""
from typing import Any

from moving_targets.util.typing import Matrix, Vector, Iteration


class Master:
    """Basic interface for a Moving Targets learner."""
    
    def __init__(self, alpha: float = 1., beta: float = 1.):
        """
        :param alpha:
            The non-negative real number which is used to calibrate the two losses in the alpha step.
        
        :param beta:
            The non-negative real number which is used to constraint the p_loss in the beta step.

        :raise `AssertionError`:
            If alpha or beta are negative.
        """
        super(Master, self).__init__()
        assert alpha >= 0, f"'alpha' should be a non-negative number, but it is {alpha}"
        assert beta >= 0, f"'beta' should be a non-negative number, but it is {beta}"

        self._alpha: float = alpha
        """The non-negative real number which is used to calibrate the two losses in the alpha step."""

        self._beta: float = beta
        """The non-negative real number which is used to constraint the p_loss in the beta step."""

    def build_model(self, macs, model, x: Matrix, y: Vector, iteration: Iteration) -> Any:
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
        raise NotImplementedError("Please implement method 'build_model'")

    def beta_step(self, macs, model, model_info: object, x: Matrix, y: Vector, iteration: Iteration) -> bool:
        """Decides whether to use or not the beta step during the current iteration.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param model:
            The inner optimization model.

        :param model_info:
            The information returned by the 'build_model' function.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param iteration:
            The current `MACS` iteration, usually a number.

        :return:
            Whether to use or not the beta step during the current iteration.
        """
        return False

    def y_loss(self, macs, model, model_info, x: Matrix, y: Vector, iteration: Iteration) -> Any:
        """Computes the loss of the model variables wrt real targets.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param model:
            The inner optimization model.

        :param model_info:
            The information returned by the 'build_model' function.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param iteration:
            The current `MACS` iteration, usually a number.

        :return:
            A real number representing the y_loss.
        """
        return 0.0

    def p_loss(self, macs, model, model_info, x: Matrix, y: Vector, iteration: Iteration) -> Any:
        """Computes the loss of the model variables wrt predictions.

        :param macs:
            Reference to the `MACS` object encapsulating the master.

        :param model:
            The inner optimization model.

        :param model_info:
            The information returned by the 'build_model' function.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param iteration:
            The current `MACS` iteration, usually a number.

        :return:
            A real number representing the p_loss.
        """
        return 0.0

    def return_solutions(self, macs, solution, model_info, x: Matrix, y: Vector, iteration: Iteration) -> Any:
        """Processes and returns the solutions given by the optimization model.

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
            Either a simple vector of adjusted targets or a tuple containing the vector and a dictionary of kwargs.
        """
        raise NotImplementedError("Please implement method 'return_solutions'")

    def adjust_targets(self, macs, x: Matrix, y: Vector, iteration: Iteration) -> Any:
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
            The vector of adjusted targets, potentially with some additional information.
        """
        raise NotImplementedError("Please implement method 'adjust_targets'")
