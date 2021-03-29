import sklearn.linear_model as lm

from src.moving_targets.learners import Learner


class LogisticRegression(Learner):
    def __init__(self, **kwargs):
        super(LogisticRegression, self).__init__()
        self.model = lm.LogisticRegression(**kwargs)

    def fit(self, macs, x, y, iteration, **kwargs):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)
