import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from moving_targets.metrics.metric import Metric


class ClassificationMetric(Metric):
    def __init__(self, metric_function, classes, name):
        super(ClassificationMetric, self).__init__(name=name)
        self.metric_function = metric_function
        self.classes = np.arange(classes) if isinstance(classes, int) else classes

    def __call__(self, x, y, p):
        if self.classes is not None:
            mask = np.in1d(y, self.classes)
            y = np.array(y)[mask]
            p = np.array(p)[mask]
        return self.metric_function(y, p)


class Precision(ClassificationMetric):
    def __init__(self, classes=None, name='precision'):
        super(Precision, self).__init__(metric_function=precision_score, classes=classes, name=name)


class Recall(ClassificationMetric):
    def __init__(self, classes=None, name='recall'):
        super(Recall, self).__init__(metric_function=recall_score, classes=classes, name=name)


class F1(ClassificationMetric):
    def __init__(self, classes=None, name='f1'):
        super(F1, self).__init__(metric_function=f1_score, classes=classes, name=name)


class Accuracy(ClassificationMetric):
    def __init__(self, classes=None, name='accuracy'):
        super(Accuracy, self).__init__(metric_function=accuracy_score, classes=classes, name=name)


class AUC(ClassificationMetric):
    def __init__(self, classes=None, name='auc'):
        super(AUC, self).__init__(metric_function=roc_auc_score, classes=classes, name=name)
