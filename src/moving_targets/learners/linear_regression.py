import sklearn.linear_model as lm

from src.moving_targets.learners import Learner


class LinearRegression(Learner):
    def __init__(self, **kwargs):
        super(LinearRegression, self).__init__()
        self.model = lm.LinearRegression(**kwargs)

    def fit(self, macs, x, y, iteration, **kwargs):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)
