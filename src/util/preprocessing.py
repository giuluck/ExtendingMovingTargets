import numpy as np
import pandas as pd
from typing import Any


class Scaler:
    def __init__(self, data: pd.DataFrame, methods: Any = 'std'):
        super(Scaler, self).__init__()
        self.translation = np.zeros_like(data.iloc[0])
        self.scaling = np.ones_like(data.iloc[0])
        if isinstance(methods, str):
            methods = {column: methods for column in data.columns}
        # compute factors
        for idx, column in enumerate(data.columns):
            method = methods.get(column)
            values = data[column]
            if method in ['std', 'standardize']:
                self.translation[idx] = values.mean()
                self.scaling[idx] = values.std()
            elif method in ['norm', 'normalize', 'minmax']:
                self.translation[idx] = values.min()
                self.scaling[idx] = values.max() - values.min()
            elif method is not None:
                raise ValueError(f'Method {method} is not supported')

    def transform(self, data):
        return (data - self.translation) / self.scaling
