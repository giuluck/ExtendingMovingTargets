import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, log_loss

from moving_targets.metrics.metric import Metric


class ClassificationMetric(Metric):
    def __init__(self, metric_function, classes, name, use_probabilities=False):
        super(ClassificationMetric, self).__init__(name=name)
        self.metric_function = metric_function
        self.classes = np.arange(classes) if isinstance(classes, int) else classes
        self.use_probabilities = use_probabilities

    def __call__(self, x, y, p):
        # HANDLE EXPLICIT CLASSES
        if self.classes is not None:
            mask = np.in1d(y, self.classes)
            y = np.array(y)[mask]
            p = np.array(p)[mask]
        # HANDLE DISCRETIZATION (BOTH BINARY AND CATEGORICAL)
        if not self.use_probabilities:
            if len(p.shape) == 1 or p.shape[1] == 1:
                p = p.round().astype(int)
            else:
                p = p.argmax(axis=1)
        # RETURN METRIC VALUE
        return self.metric_function(y, p)


class CrossEntropy(ClassificationMetric):
    def __init__(self, classes=None, eps=1e-3, name='crossentropy'):
        def crossentropy(y_true, y_pred):
            return log_loss(y_true, y_pred, eps=eps)

        super(CrossEntropy, self).__init__(metric_function=crossentropy, classes=classes, name=name)


class Precision(ClassificationMetric):
    def __init__(self, classes=None, name='precision'):
        super(Precision, self).__init__(metric_function=precision_score, classes=classes, name=name)


class Recall(ClassificationMetric):
    def __init__(self, classes=None, name='recall'):
        super(Recall, self).__init__(metric_function=recall_score, classes=classes, name=name)


class F1(ClassificationMetric):
    def __init__(self, classes=None, name='f1_score'):
        super(F1, self).__init__(metric_function=f1_score, classes=classes, name=name)


class Accuracy(ClassificationMetric):
    def __init__(self, classes=None, name='accuracy'):
        super(Accuracy, self).__init__(metric_function=accuracy_score, classes=classes, name=name)


class AUC(ClassificationMetric):
    def __init__(self, classes=None, name='auc'):
        super(AUC, self).__init__(metric_function=roc_auc_score, classes=classes, use_probabilities=True, name=name)
