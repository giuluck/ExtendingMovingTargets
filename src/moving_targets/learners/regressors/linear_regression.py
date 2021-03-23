import sklearn.linear_model as lm

from src.moving_targets.learners.regressors import Regressor


class LinearRegression(Regressor):
    def __init__(self, **kwargs):
        super(LinearRegression, self).__init__()
        self.model = lm.LinearRegression(**kwargs)

    def fit(self, x, y, **kwargs):
        self.model.fit(x, y, **kwargs)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        raise NotImplementedError('')
