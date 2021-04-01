import numpy as np

from src.moving_targets.metrics import Metric


class ClassFrequenciesStd(Metric):
    def __init__(self, classes=None, name='class frequencies std'):
        super(ClassFrequenciesStd, self).__init__(name)
        self.classes = np.arange(classes) if isinstance(classes, int) else classes

    def __call__(self, x, y, p):
        # bincount is similar to np.unique(..., return_counts=True) but allows to fix a minimum number of classes
        # in this way, if the predictions are all the same, the counts will be [n, 0, ..., 0] instead of [n]
        minlength = 1 + max(np.max(self.classes), np.max(y), np.max(p))
        classes_counts = np.bincount(p, minlength=minlength)
        classes_counts = classes_counts if self.classes is None else classes_counts[self.classes]
        return np.std(classes_counts / len(p))
