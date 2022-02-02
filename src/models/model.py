from typing import Any

import numpy as np
import pandas as pd
from moving_targets.util.errors import not_implemented_message


class Model:
    """Abstract Model Interface."""

    __name__: str = 'Model'

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> Any:
        """Fits the model."""
        raise NotImplementedError(not_implemented_message('fit'))

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Uses the model for target prediction."""
        raise NotImplementedError(not_implemented_message('predict'))
