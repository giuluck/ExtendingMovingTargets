from typing import Optional, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Scaler:
    def __init__(self, methods: Optional[str] = 'std'):
        super(Scaler, self).__init__()
        self.methods: Optional[str] = methods
        self.translation: Optional[np.ndarray] = None
        self.scaling: Optional[np.ndarray] = None

    def fit(self, data: object):
        # handle non-pandas data
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(np.array(data))

        # handle all-the-same methods
        if isinstance(self.methods, dict):
            methods = self.methods
        else:
            methods = {column: self.methods for column in data.columns}

        # default values (translation = 0, scaling = 1)
        self.translation = np.zeros(len(data.iloc[0]))
        self.scaling = np.ones(len(data.iloc[0]))

        # compute factors
        for idx, column in enumerate(data.columns):
            method = methods.get(column)
            values = data[column].values
            if method in ['std', 'standardize']:
                self.translation[idx] = values.mean()
                self.scaling[idx] = values.std()
            elif method in ['norm', 'normalize', 'minmax']:
                self.translation[idx] = values.min()
                self.scaling[idx] = values.max() - values.min()
            elif method in ['zero', 'max', 'zeromax']:
                self.translation[idx] = 0.0
                self.scaling[idx] = values.max()
            elif isinstance(method, tuple):
                minimum, maximum = method
                self.translation[idx] = minimum
                self.scaling[idx] = maximum - minimum
            elif method is not None:
                raise ValueError(f'Method {method} is not supported')
        return self

    def transform(self, data: object):
        return (data - self.translation) / self.scaling

    def fit_transform(self, data: object):
        return self.fit(data).transform(data)

    def inverse_transform(self, data: object):
        return (data * self.scaling) + self.translation

    @staticmethod
    def get_default(num_features: int) -> object:
        return Scaler(methods=None).fit(data=[[0.] * num_features])


def split_dataset(*arg, test_size: float = 0.2, val_size: Optional[float] = None, extrapolation: object = None, **kwargs):
    # handle default values
    val_size = test_size if val_size is None else val_size
    val_size = val_size if isinstance(val_size, float) else val_size / len(arg[0])
    test_size = test_size if isinstance(test_size, float) else test_size / len(arg[0])
    # split train/test
    if extrapolation is None:
        splits = train_test_split(*arg, test_size=test_size, **kwargs)
    else:
        x = arg[0]
        if not isinstance(extrapolation, dict):
            extrapolation = {col: extrapolation for col in x.columns}
        train_mask, test_mask = np.ones(len(x)).astype(bool), np.ones(len(x)).astype(bool)
        # removes all the data points at the borders for each feature (lq and uq are the quantile values for test set)
        for col, ex in extrapolation.items():
            feat = x[col]
            lq, uq = ex / 2, 1 - ex / 2 if isinstance(ex, float) else ex
            train_mask = np.logical_and(train_mask, np.logical_and(feat > feat.quantile(lq), feat < feat.quantile(uq)))
            test_mask = np.logical_and(test_mask, np.logical_or(feat <= feat.quantile(lq), feat >= feat.quantile(uq)))
        splits = []
        # create the splits from the initial data by appending the train and the test partition for each vector
        for a in arg:
            splits.append(a[train_mask])
            splits.append(a[test_mask])
    train_data, test_data = splits[::2], (splits[1] if len(arg) == 1 else splits[1::2])
    # split val/test only if necessary
    if val_size == 0.0:
        return {'train': train_data, 'test': test_data}
    else:
        splits = train_test_split(*train_data, test_size=val_size, **kwargs)
        train_data, val_data = splits if len(arg) == 1 else (splits[::2], splits[1::2])
        return {'train': train_data, 'validation': val_data, 'test': test_data}
