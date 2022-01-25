from typing import Any

import numpy as np
from moving_targets.util.errors import not_implemented_message


class Model:
    """Abstract Model Interface."""

    def fit(self, x, y: np.ndarray) -> Any:
        """Fits the model."""
        raise NotImplementedError(not_implemented_message('fit'))

    def predict(self, x) -> np.ndarray:
        """Uses the model for target prediction."""
        raise NotImplementedError(not_implemented_message('predict'))
