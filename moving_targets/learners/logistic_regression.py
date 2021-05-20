from typing import Any

import numpy as np
import sklearn.linear_model as lm

from moving_targets.learners.learner import Learner


class LogisticRegression(Learner):
    def __init__(self, **kwargs):
        super(LogisticRegression, self).__init__()
        self.model = lm.LogisticRegression(**kwargs)

    def fit(self, macs, x: np.ndarray, y, iteration: Any, **kwargs):
        self.model.fit(x, y)

    def predict(self, x) -> np.ndarray:
        return self.model.predict(x)

    def predict_proba(self, x) -> np.ndarray:
        return self.model.predict_proba(x)
