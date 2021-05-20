from typing import Any

import numpy as np


class Learner:
    def __init__(self):
        super(Learner, self).__init__()

    def fit(self, macs, x, y, iteration: Any, **kwargs):
        pass

    def predict(self, x) -> np.ndarray:
        pass
