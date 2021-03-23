from sklearn.metrics import accuracy_score

from src.moving_targets.metrics import Metric


class Accuracy(Metric):
    def __init__(self, name='accuracy'):
        super(Accuracy, self).__init__(name)

    def __call__(self, x, y, pred):
        return accuracy_score(y, pred)
