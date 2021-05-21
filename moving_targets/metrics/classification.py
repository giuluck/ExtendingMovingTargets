"""Classification Metrics."""

import numpy as np
from typing import Callable, Optional
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, log_loss

from moving_targets.metrics.metric import Metric
from moving_targets.utils.typing import Classes, Matrix, Vector


class ClassificationMetric(Metric):
    """Basic interface for a Moving Target's classification metric.

    Args:
        metric_function: callable function that computes the metric given the true targets and the predictions.
        classes: the number of classes or a vector of class labels.
        name: the name of the metric.
        use_prob: whether to use the output labels or the output probabilities.
    """

    def __init__(self, metric_function: Callable, classes: Optional[Classes], name: str, use_prob: bool = False):
        super(ClassificationMetric, self).__init__(name=name)
        self.metric_function: Callable = metric_function
        self.classes: np.ndarray = np.arange(classes) if isinstance(classes, int) else classes
        self.use_prob: bool = use_prob

    def __call__(self, x: Matrix, y: Vector, p: Vector) -> float:
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
    """Negative Log-Likelihood Loss.

    Args:
        clip_value = the clipping value to be used to avoid numerical errors with the log.
        classes: the number of classes or a vector of class labels.
        name: the name of the metric.
    """

    def __init__(self, clip_value: float = 1e-15, classes: Optional[Classes] = None, name: str = 'crossentropy'):
        def _crossentropy(y_true, y_pred):
            return log_loss(y_true, y_pred, eps=clip_value)

        super(CrossEntropy, self).__init__(metric_function=_crossentropy, classes=classes, name=name, use_prob=False)


class Precision(ClassificationMetric):
    """Precision Score.

    Args:
        classes: the number of classes or a vector of class labels.
        name: the name of the metric.
    """

    def __init__(self, classes: Optional[Classes] = None, name: str = 'precision'):
        super(Precision, self).__init__(metric_function=precision_score, classes=classes, name=name, use_prob=False)


class Recall(ClassificationMetric):
    """Recall Score.

    Args:
        classes: the number of classes or a vector of class labels.
        name: the name of the metric.
    """

    def __init__(self, classes: Optional[Classes] = None, name: str = 'recall'):
        super(Recall, self).__init__(metric_function=recall_score, classes=classes, name=name)


class F1(ClassificationMetric):
    """F1 Score.

    Args:
        classes: the number of classes or a vector of class labels.
        name: the name of the metric.
    """

    def __init__(self, classes: Optional[Classes] = None, name: str = 'f1_score'):
        super(F1, self).__init__(metric_function=f1_score, classes=classes, name=name)


class Accuracy(ClassificationMetric):
    """Accuracy Score.

    Args:
        classes: the number of classes or a vector of class labels.
        name: the name of the metric.
    """

    def __init__(self, classes: Optional[Classes] = None, name: str = 'accuracy'):
        super(Accuracy, self).__init__(metric_function=accuracy_score, classes=classes, name=name)


class AUC(ClassificationMetric):
    """AUC Score.

    Args:
        classes: the number of classes or a vector of class labels.
        name: the name of the metric.
    """

    def __init__(self, classes: Optional[Classes] = None, name: str = 'auc'):
        super(AUC, self).__init__(metric_function=roc_auc_score, classes=classes, name=name, use_prob=True)
