import numpy as np
from sklearn.metrics import accuracy_score

from moving_targets.metrics.metric import Metric


class Accuracy(Metric):
    def __init__(self, classes=None, name='accuracy'):
        super(Accuracy, self).__init__(name=name)
        self.classes = np.arange(classes) if isinstance(classes, int) else classes

    def __call__(self, x, y, p):
        if self.classes is not None:
            mask = np.in1d(y, self.classes)
            y = np.array(y)[mask]
            p = np.array(p)[mask]
        return accuracy_score(y, p)
