from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from moving_targets.metrics.metric import Metric


class RegressionMetric(Metric):
    def __init__(self, metric_function, name):
        super(RegressionMetric, self).__init__(name=name)
        self.metric_function = metric_function

    def __call__(self, x, y, p):
        return self.metric_function(y, p)


class MSE(RegressionMetric):
    def __init__(self, name='mse'):
        super(MSE, self).__init__(metric_function=mean_squared_error, name=name)


class MAE(RegressionMetric):
    def __init__(self, name='mae'):
        super(MAE, self).__init__(metric_function=mean_absolute_error, name=name)


class R2(RegressionMetric):
    def __init__(self, name='r2'):
        super(R2, self).__init__(metric_function=r2_score, name=name)
