import numpy as np
from sklearn.metrics import roc_auc_score

from moving_targets.metrics.metric import Metric


class AUC(Metric):
    def __init__(self, classes=None, name='AUC'):
        super(AUC, self).__init__(name=name)
        self.classes = np.arange(classes) if isinstance(classes, int) else classes

    def __call__(self, x, y, p):
        if self.classes is not None:
            mask = np.in1d(y, self.classes)
            y = np.array(y)[mask]
            p = np.array(p)[mask]
        return roc_auc_score(y, p)
