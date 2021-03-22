import numpy as np
from sklearn.metrics import accuracy_score

class Metric:
    def __init__(self, name):
        super(Metric, self).__init__()
        self.name = name

    def __call__(self, x, y, pred):
        pass


class MultiMetric(Metric):
    def __init__(self, metrics):
        super(MultiMetric, self).__init__(name=None)
        self.metrics = metrics

    def __call__(self, x, y, pred):
        return {metric.name: metric(x, y, pred) for metric in self.metrics}


class Accuracy(Metric):
    def __init__(self, name='accuracy'):
        super(Accuracy, self).__init__(name)

    def __call__(self, x, y, pred):
        return accuracy_score(y, pred)


class ClassFrequenciesStd(Metric):
    def __init__(self, num_classes=None, name='class frequencies std'):
        super(ClassFrequenciesStd, self).__init__(name)
        self.num_classes = num_classes

    def __call__(self, x, y, pred):
        # bincount is similar to np.unique(..., return_counts=True) but allows to fix a minimum number of classes
        # in this way, if the predictions are all the same, the counts will be [n, 0, ..., 0] instead of [n]
        if self.num_classes is None:
            num_classes = max(np.argmax(y), np.argmax(pred))
        else:
            num_classes = self.num_classes
        classes_counts = np.bincount(pred, minlength=num_classes)
        return np.std(classes_counts / len(pred))
