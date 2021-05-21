"""Basic Master Interface."""

from moving_targets.utils.typing import Matrix, Vector, Iteration


class Master:
    """Basic interface for a Moving Target's learner.

    Args:
        alpha: the non-negative real number which is used to calibrate the two losses in the alpha step.
        beta: the non-negative real number which is used to constraint the p_loss in the beta step.

    Raises:
        `AssertionError` if alpha or beta are negative.
    """

    def __init__(self, alpha: float = 1., beta: float = 1.):
        super(Master, self).__init__()
        assert alpha >= 0, "alpha should be a non-negative number"
        assert beta >= 0, "beta should be a non-negative number"
        self.alpha: float = alpha
        self.beta: float = beta

    def build_model(self, macs, model, x: Matrix, y: Vector, iteration: Iteration) -> object:
        """Creates the model variables and adds the problem constraints.

        Args:
            macs: reference to the MACS object encapsulating the master.
            model: the inner optimization model.
            x: the matrix/dataframe of training samples.
            y: the vector of training labels.
            iteration: the current MACS iteration, usually a number.

        Returns:
            Any object containing information that may be useful in other methods.
        """
        raise NotImplementedError("Please implement method 'build_model'")

    def beta_step(self, macs, model, model_info: object, x: Matrix, y: Vector, iteration: Iteration) -> bool:
        """Decides whether to use or not the beta step during the current iteration.

        Args:
            macs: reference to the MACS object encapsulating the master.
            model: the inner optimization model.
            model_info: the information returned by the 'build_model' function.
            x: the matrix/dataframe of training samples.
            y: the vector of training labels.
            iteration: the current MACS iteration, usually a number.

        Returns:
            Whether to use or not the beta step during the current iteration.
        """
        return False

    def y_loss(self, macs, model, model_info, x: Matrix, y: Vector, iteration: Iteration) -> float:
        """Computes the loss of the model variables wrt real targets.

        Args:
            macs: reference to the MACS object encapsulating the master.
            model: the inner optimization model.
            model_info: the information returned by the 'build_model' function.
            x: the matrix/dataframe of training samples.
            y: the vector of training labels.
            iteration: the current MACS iteration, usually a number.

        Returns:
            A real number representing the y_loss.
        """
        return 0.0

    def p_loss(self, macs, model, model_info, x: Matrix, y: Vector, iteration: Iteration) -> float:
        """Computes the loss of the model variables wrt predictions.

        Args:
            macs: reference to the MACS object encapsulating the master.
            model: the inner optimization model.
            model_info: the information returned by the 'build_model' function.
            x: the matrix/dataframe of training samples.
            y: the vector of training labels.
            iteration: the current MACS iteration, usually a number.

        Returns:
            A real number representing the p_loss.
        """
        return 0.0

    def return_solutions(self, macs, solution, model_info, x: Matrix, y: Vector, iteration: Iteration) -> object:
        """Processes and returns the solutions given by the optimization model.

        Args:
            macs: reference to the MACS object encapsulating the master.
            solution: the object containing information about the solution of the problem.
            model_info: the information returned by the 'build_model' function.
            x: the matrix/dataframe of training samples.
            y: the vector of training labels.
            iteration: the current MACS iteration, usually a number.

        Returns:
            Either a simple vector of adjusted targets or a tuple containing the vector and a dictionary of kwargs.
        """
        raise NotImplementedError("Please implement method 'return_solutions'")

    def adjust_targets(self, macs, x: Matrix, y: Vector, iteration: Iteration) -> object:
        """Core function of the Master object. Builds the model dependently on the kind solver and returns the adjusted
        targets.

        Args:
            macs: reference to the MACS object encapsulating the master.
            x: the matrix/dataframe of training samples.
            y: the vector of training labels.
            iteration: the current MACS iteration, usually a number.

        Returns:
            Either a simple vector of adjusted targets or a tuple containing the vector and a dictionary of kwargs.
        """
        raise NotImplementedError("Please implement method 'adjust_targets'")
