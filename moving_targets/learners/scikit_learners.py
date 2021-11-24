"""Sklearn-based Learners."""

import sklearn.linear_model as lm

from moving_targets.learners.learner import Learner
from moving_targets.util.typing import Matrix, Vector, Iteration


class ScikitLearner(Learner):
    """Wrapper for a custom Scikit-Learn model."""

    def __init__(self, model):
        """
        :param model:
            The Scikit-Learn model.
        """
        super(ScikitLearner, self).__init__()

        self.model = model
        """The Scikit-Learn model."""

    def fit(self, macs, x: Matrix, y: Vector, iteration: Iteration, **additional_kwargs):
        self.model.fit(x, y)

    def predict(self, x: Matrix) -> Vector:
        """Predicts the labels of the given input samples.

        :param x:
            The input samples.

        :return:
            The predicted labels.
        """
        return self.model.predict(x)


class ScikitRegressor(ScikitLearner):
    """Wrapper for a custom Scikit-Learn regression model."""


class ScikitClassifier(ScikitLearner):
    """Wrapper for a custom Scikit-Learn classification model."""

    def predict_proba(self, x: Matrix) -> Vector:
        """Predicts the probabilities for each class of the given input samples.

        :param x:
            The input samples.

        :return:
            The predicted probabilities.
        """
        return self.model.predict_proba(x)


class LinearRegression(ScikitLearner):
    """Scikit-Learn Linear Regression wrapper."""

    def __init__(self, **scikit_kwargs):
        """
        :param scikit_kwargs:
            Custom arguments to be passed to a sklearn.linear_model.LogisticRegression instance.
        """
        super(LinearRegression, self).__init__(lm.LinearRegression(**scikit_kwargs))


class LogisticRegression(ScikitClassifier):
    """Scikit-Learn Logistic Regression wrapper."""

    def __init__(self, **scikit_kwargs):
        """
        :param scikit_kwargs:
            Custom arguments to be passed to a sklearn.linear_model.LogisticRegression instance.
        """
        super(LogisticRegression, self).__init__(lm.LogisticRegression(**scikit_kwargs))
