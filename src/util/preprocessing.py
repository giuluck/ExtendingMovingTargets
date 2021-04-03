import numpy as np
import pandas as pd


class Scaler:
    def __init__(self, data, methods='std'):
        super(Scaler, self).__init__()
        # handle non-pandas data
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(np.array(data), columns=['data'])
        # handle all-the-same methods
        if isinstance(methods, str):
            methods = {column: methods for column in data.columns}
        # default values (translation = 0, scaling = 1)
        self.translation = np.zeros_like(data.iloc[0])
        self.scaling = np.ones_like(data.iloc[0])
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

    def invert(self, data):
        return (data * self.scaling) + self.translation
