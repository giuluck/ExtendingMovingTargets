"""Basic Learner Interface."""

from moving_targets.util.typing import Vector, Iteration, Matrix


class Learner:
    """Basic interface for a Moving Target's learner."""

    def __init__(self):
        super(Learner, self).__init__()

    def fit(self, macs, x: Matrix, y: Vector, iteration: Iteration, **kwargs):
        """Fits the learner according to the implemented procedure using (x, y) as training data.

        Args:
            macs: reference to the MACS object encapsulating the learner.
            x: the matrix/dataframe of training samples.
            y: the vector of training labels.
            iteration: the current MACS iteration, usually a number.
            **kwargs: any other useful parameter.
        """
        raise NotImplementedError("Please implement method 'fit'")

    def predict(self, x: Matrix) -> Vector:
        """Uses the fitted learner configuration to predict labels from input samples.

        Args:
            x: the matrix/dataframe of input samples.

        Returns:
            The vector of predicted labels.
        """
        raise NotImplementedError("Please implement method 'predict'")
