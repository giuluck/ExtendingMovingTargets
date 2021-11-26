"""Basic Learner Interface."""
from abc import ABC
from typing import Union

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

    def predict(self, x: Matrix) -> Union[Vector, Matrix]:
        """Uses the fitted learner configuration to predict labels from input samples.

        :param x:
            The matrix/dataframe of input samples.

        :return:
            The vector of predicted labels.
        """
        raise NotImplementedError("Please implement method 'predict'")


class Classifier(Learner, ABC):
    """Basic interface for a Moving Targets classifier."""

    @staticmethod
    def get_classes(prob: Matrix) -> Union[Vector, Matrix]:
        """Gets the output classes given the output probabilities per class.

        :param prob:
            The output probabilities of a `Classifier`.

        :return:
            The respective output classes.
        """
        # strategy varies depending on binary vs. multiclass classification
        if len(prob.shape) == 1 or prob.shape[1] == 1:
            return prob.round().astype(int)
        else:
            return prob.argmax(axis=1)

    def predict_classes(self, x: Matrix) -> Vector:
        """Predicts the output classes of the given input samples.

        :param x:
            The input samples.

        :return:
            The predicted classes.
        """
        return Classifier.get_classes(self.predict(x))