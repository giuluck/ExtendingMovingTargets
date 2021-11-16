"""Classification Metrics."""

from typing import Callable, Optional

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, log_loss

from moving_targets.metrics.metric import Metric
from moving_targets.util.typing import Classes, Matrix, Vector


class ClassificationMetric(Metric):
    """Basic interface for a Moving Targets classification metric."""

    def __init__(self, metric_function: Callable, classes: Optional[Classes], name: str, use_prob: bool = False):
        """
        :param metric_function:
            Callable function that computes the metric given the true targets and the predictions.

        :param classes:
            The number of classes or a vector of class labels.

        :param name:
            The name of the metric.

        :param use_prob:
            Whether to use the output labels or the output probabilities.
        """
        super(ClassificationMetric, self).__init__(name=name)

        self.metric_function: Callable = metric_function
        """Callable function that computes the metric given the true targets and the predictions."""

        self.classes: np.ndarray = np.arange(classes) if isinstance(classes, int) else classes
        """The number of classes or a vector of class labels."""

        self.use_prob: bool = use_prob
        """Whether to use the output labels or the output probabilities."""

    def __call__(self, x: Matrix, y: Vector, p: Vector) -> float:
        """Core method used to compute the metric value.

        :param x:
            The input matrix (unused).

        :param y:
            The vector of ground truths.

        :param p:
            The vector of predictions.

        :returns:
            The metric value.
        """
        # HANDLE EXPLICIT CLASSES
        if self.classes is not None:
            mask = np.in1d(y, self.classes)
            y = np.array(y)[mask]
            p = np.array(p)[mask]
        # HANDLE DISCRETIZATION (BOTH BINARY AND CATEGORICAL)
        if not self.use_prob:
            if len(p.shape) == 1 or p.shape[1] == 1:
                p = p.round().astype(int)
            else:
                p = p.argmax(axis=1)
        # RETURN METRIC VALUE
        return self.metric_function(y, p)


class CrossEntropy(ClassificationMetric):
    """Negative Log-Likelihood Loss."""

    def __init__(self, clip_value: float = 1e-15, classes: Optional[Classes] = None, name: str = 'crossentropy'):
        """
        :param clip_value:
            The clipping value to be used to avoid numerical errors with the log.

        :param classes:
            The number of classes or a vector of class labels.

        :param name:
            The name of the metric.
        """
        super(CrossEntropy, self).__init__(
            metric_function=lambda y_true, y_pred: log_loss(y_true, y_pred, eps=clip_value),
            classes=classes,
            name=name,
            use_prob=False
        )


class Precision(ClassificationMetric):
    """Precision Score."""

    def __init__(self, classes: Optional[Classes] = None, name: str = 'precision'):
        """
        :param classes:
            The number of classes or a vector of class labels.

        :param name:
            The name of the metric.
        """
        super(Precision, self).__init__(metric_function=precision_score, classes=classes, name=name, use_prob=False)


class Recall(ClassificationMetric):
    """Recall Score."""

    def __init__(self, classes: Optional[Classes] = None, name: str = 'recall'):
        """
        :param classes:
            The number of classes or a vector of class labels.

        :param name:
            The name of the metric.
        """
        super(Recall, self).__init__(metric_function=recall_score, classes=classes, name=name, use_prob=False)


class F1(ClassificationMetric):
    """F1 Score."""

    def __init__(self, classes: Optional[Classes] = None, name: str = 'f1_score'):
        """
        :param classes:
            The number of classes or a vector of class labels.

        :param name:
            The name of the metric.
        """
        super(F1, self).__init__(metric_function=f1_score, classes=classes, name=name, use_prob=False)


class Accuracy(ClassificationMetric):
    """Accuracy Score."""

    def __init__(self, classes: Optional[Classes] = None, name: str = 'accuracy'):
        """
        :param classes:
            The number of classes or a vector of class labels.

        :param name:
            The name of the metric.
        """
        super(Accuracy, self).__init__(metric_function=accuracy_score, classes=classes, name=name, use_prob=False)


class AUC(ClassificationMetric):
    """AUC Score."""

    def __init__(self, classes: Optional[Classes] = None, name: str = 'auc'):
        """
        :param classes:
            The number of classes or a vector of class labels.

        :param name:
            The name of the metric.
        """
        super(AUC, self).__init__(metric_function=roc_auc_score, classes=classes, name=name, use_prob=True)
