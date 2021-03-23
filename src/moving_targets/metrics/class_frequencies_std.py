import numpy as np

from src.moving_targets.metrics import Metric


class ClassFrequenciesStd(Metric):
    def __init__(self, num_classes=None, name='class frequencies std'):
        super(ClassFrequenciesStd, self).__init__(name)
        self.num_classes = num_classes

    def __call__(self, x, y, pred):
        # bincount is similar to np.unique(..., return_counts=True) but allows to fix a minimum number of classes
        # in this way, if the predictions are all the same, the counts will be [n, 0, ..., 0] instead of [n]
        minlength = max(self.num_classes, np.max(y), np.max(pred))
        classes_counts = np.bincount(pred, minlength=minlength)
        return np.std(classes_counts / len(pred))
