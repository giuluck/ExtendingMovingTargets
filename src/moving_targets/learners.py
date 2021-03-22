import sklearn.linear_model as lm

class Learner:
    def __init__(self):
        super(Learner, self).__init__()

    def fit(self, x, y, **kwargs):
        pass

    def predict(self, x):
        pass

    def predict_proba(self, x):
        pass


class LinearRegression(Learner):
    def __init__(self, **kwargs):
        super(LinearRegression, self).__init__()
        self.model = lm.LinearRegression(**kwargs)

    def fit(self, x, y, **kwargs):
        self.model.fit(x, y, **kwargs)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        raise NotImplementedError('Regression learners cannot return probabilities')


class LogisticRegression(Learner):
    def __init__(self, **kwargs):
        super(LogisticRegression, self).__init__()
        self.model = lm.LogisticRegression(**kwargs)

    def fit(self, x, y, **kwargs):
        self.model.fit(x, y, **kwargs)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)
