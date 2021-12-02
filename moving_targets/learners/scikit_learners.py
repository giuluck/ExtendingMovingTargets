"""Sklearn-based Learners."""
from typing import Any

import sklearn.linear_model as lm

from moving_targets.learners.learner import Learner, Classifier
from moving_targets.util.typing import Iteration


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

    def fit(self, macs, x, y, iteration: Iteration, **additional_kwargs):
        self.model.fit(x, y)

    def predict(self, x) -> Any:
        """Predicts the labels of the given input samples.

        :param x:
            The input samples.

        :return:
            The predicted labels.
        """
        return self.model.predict(x)


class ScikitClassifier(ScikitLearner, Classifier):
    """Wrapper for a custom Scikit-Learn classification model."""

    def predict(self, x) -> Any:
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
