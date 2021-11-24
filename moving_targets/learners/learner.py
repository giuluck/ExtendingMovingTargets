"""Basic Learner Interface."""

from moving_targets.util.typing import Vector, Iteration, Matrix


class Learner:
    """Basic interface for a Moving Targets learner."""

    def __init__(self):
        """"""
        super(Learner, self).__init__()

    def fit(self, macs, x: Matrix, y: Vector, iteration: Iteration, **additional_kwargs):
        """Fits the learner according to the implemented procedure using (x, y) as training data.

        :param macs:
            Reference to the `MACS` object encapsulating the learner.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param iteration:
            The current `MACS` iteration, usually a number.

        :param additional_kwargs:
            Any other useful parameter.
        """
        raise NotImplementedError("Please implement method 'fit'")

    def predict(self, x: Matrix) -> Vector:
        """Uses the fitted learner configuration to predict labels from input samples.

        :param x:
            The matrix/dataframe of input samples.

        :return:
            The vector of predicted labels.
        """
        raise NotImplementedError("Please implement method 'predict'")
