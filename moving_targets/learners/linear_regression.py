from typing import Any

import numpy as np
import sklearn.linear_model as lm

from moving_targets.learners.learner import Learner


class LinearRegression(Learner):
    def __init__(self, **kwargs):
        super(LinearRegression, self).__init__()
        self.model = lm.LinearRegression(**kwargs)

    def fit(self, macs, x, y, iteration: Any, **kwargs):
        self.model.fit(x, y)

    def predict(self, x) -> np.ndarray:
        return self.model.predict(x)
