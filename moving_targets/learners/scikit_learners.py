"""Sklearn-based Learners."""

import sklearn.linear_model as lm

from moving_targets.learners.learner import Learner
from moving_targets.utils.typing import Matrix, Vector, Iteration


class ScikitLearner(Learner):
    """Wrapper for a custom Scikit-Learn model.

    Args:
        model: the Scikit-Learn model.
    """

    def __init__(self, model):
        super(ScikitLearner, self).__init__()
        self.model = model

    # noinspection PyMissingOrEmptyDocstring
    def fit(self, macs, x: Matrix, y: Vector, iteration: Iteration, **kwargs):
        self.model.fit(x, y)

    # noinspection PyMissingOrEmptyDocstring
    def predict(self, x: Matrix) -> Vector:
        return self.model.predict(x)


class ScikitRegressor(ScikitLearner):
    """Wrapper for a custom Scikit-Learn regression model."""


class ScikitClassifier(ScikitLearner):
    """Wrapper for a custom Scikit-Learn classification model."""

    def predict_proba(self, x: Matrix) -> Vector:
        """Uses the fitted learner configuration to predict the probabilities for each class from input samples.

        Args:
            x: the matrix/dataframe of input samples.

        Returns:
            The vector of predicted probabilities for each class.
        """
        return self.model.predict_proba(x)


class LinearRegression(ScikitLearner):
    """Scikit-Learn Linear Regression wrapper.

    Args:
        **kwargs: custom arguments to be passed to a sklearn.linear_model.LogisticRegression instance.
    """

    def __init__(self, **kwargs):
        super(LinearRegression, self).__init__(lm.LinearRegression(**kwargs))


class LogisticRegression(ScikitClassifier):
    """Scikit-Learn Logistic Regression wrapper.

    Args:
        **kwargs: custom arguments to be passed to a sklearn.linear_model.LogisticRegression instance.
    """

    def __init__(self, **kwargs):
        super(LogisticRegression, self).__init__(lm.LogisticRegression(**kwargs))
