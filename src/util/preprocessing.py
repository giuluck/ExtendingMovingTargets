import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Scaler:
    def __init__(self, data, methods: object = 'std'):
        super(Scaler, self).__init__()
        # handle non-pandas data
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(np.array(data), columns=['data'])
        # handle all-the-same methods
        if isinstance(methods, str) or methods is None:
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
            elif method in ['zero', 'max', 'zeromax']:
                self.translation[idx] = 0.0
                self.scaling[idx] = values.max()
            elif method is not None:
                raise ValueError(f'Method {method} is not supported')

    def transform(self, data):
        return (data - self.translation) / self.scaling

    def invert(self, data):
        return (data * self.scaling) + self.translation

    @staticmethod
    def get_default(num_features):
        return Scaler([0.] * num_features, methods=None)


def split_dataset(*data, test_size=0.2, val_size=None, **kwargs):
    # handle default values
    val_size = test_size if val_size is None else val_size
    val_size = val_size if isinstance(val_size, float) else val_size / len(data[0])
    test_size = test_size if isinstance(test_size, float) else test_size / len(data[0])
    # split train/test
    splits = train_test_split(*data, test_size=test_size, **kwargs)
    train_data, test_data = splits[::2], (splits[1] if len(data) == 1 else splits[1::2])
    # split val/test only if necessary
    if val_size == 0.0:
        return train_data, test_data
    else:
        splits = train_test_split(*train_data, test_size=val_size, **kwargs)
        train_data, val_data = splits if len(data) == 1 else (splits[::2], splits[1::2])
        return train_data, val_data, test_data
