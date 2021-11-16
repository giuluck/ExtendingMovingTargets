"""Processing utils."""

from typing import Union, Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from moving_targets.util.typing import Matrix, Vector
from src.util.typing import Methods, Extrapolation


class Scaler:
    """Custom scaler that is able to treat each feature separately."""

    def __init__(self, methods: Methods = 'std'):
        """
        :param methods:
            Either a string/tuple, a list of strings/tuples, or a dictionary of strings/tuples.

            Each string/tuple represents a method:

            - 'std', or 'standardize', standardizes the feature
            - 'norm', or 'normalize', or 'minmax', normalizes the feature from (min, max) into (0, 1)
            - 'zero', or 'max', or 'zeromax', normalizes the feature from (0, max) into (0, 1)
            - Tuple[Number, Number], normalizes the feature from (t1, t2) into (0, 1)
            - None, performs no scaling

            If a single value is passed, all the features are scaled with the same method.
            If a list or a dictionary is passed, each feature is scaled with the respective method.
        """
        super(Scaler, self).__init__()

        self.methods = methods
        """The scaling methods."""

        self._translation: Optional[np.ndarray] = None
        """The translation vector."""

        self._scaling: Optional[np.ndarray] = None
        """The scaling vector."""

    def fit(self, data: Matrix):
        """Fits the scaler parameters.

        :param data:
            The matrix/dataframe of samples.

        :returns:
            The scaler itself.
        """
        # handle non-pandas data
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(np.array(data))

        # handle all-the-same methods
        if isinstance(self.methods, dict):
            methods = self.methods
        else:
            methods = {column: self.methods for column in data.columns}

        # default values (translation = 0, scaling = 1)
        self._translation = np.zeros(len(data.iloc[0]))
        self._scaling = np.ones(len(data.iloc[0]))

        # compute factors
        for idx, column in enumerate(data.columns):
            method = methods.get(column)
            values = data[column].values
            if method in ['std', 'standardize']:
                self._translation[idx] = values.mean()
                self._scaling[idx] = values.std()
            elif method in ['norm', 'normalize', 'minmax']:
                self._translation[idx] = values.min()
                self._scaling[idx] = values.max() - values.min()
            elif method in ['zero', 'max', 'zeromax']:
                self._translation[idx] = 0.0
                self._scaling[idx] = values.max()
            elif isinstance(method, tuple):
                minimum, maximum = method
                self._translation[idx] = minimum
                self._scaling[idx] = maximum - minimum
            elif method is not None:
                raise ValueError(f'Method {method} is not supported')
        self._scaling[self._scaling == 0] = 1.0  # handle case with null scaling factor
        return self

    def transform(self, data: Matrix) -> Matrix:
        """Transforms the data according to the scaler parameters.

        :param data:
            The matrix/dataframe of samples.

        :returns:
            The scaled data.
        """
        return (data - self._translation) / self._scaling

    def fit_transform(self, data: Matrix) -> Matrix:
        """Fits the scaler parameters.

        :param data:
            The matrix/dataframe of samples.

        :returns:
            The scaled data.
        """
        return self.fit(data).transform(data)

    def inverse_transform(self, data: Matrix) -> Matrix:
        """Inverts the scaling according to the scaler parameters.

        :param data:
            The previously scaled matrix/dataframe of samples.

        :returns:
            The original data.
        """
        return (data * self._scaling) + self._translation

    @staticmethod
    def get_default(num_features: int) -> Any:
        """Builds a blank scaler which works on data with the given number of features.

        :param num_features:
            The number of features.

        :returns:
            A blank scaler.
        """
        return Scaler(methods=None).fit(data=[[0.] * num_features])


Scalers = Union[Optional[Scaler], Tuple[Optional[Scaler], Optional[Scaler]]]
"""Either an (optional) scaler for both the input and the output data, or a tuple of (optional) scalers for the input
and the output data, respectively."""

SplitArgs = Union[Matrix, Vector]
"""Either a `Matrix` or a `Vector` that needs to be split into train/test."""

ValidationArgs = Union[pd.DataFrame, pd.Series]
"""Either a `pandas.DataFrame` or a `pandas.Series` that needs to be split for cross-validation."""


def split_dataset(*dataset: SplitArgs,
                  test_size: Union[int, float] = 0.2,
                  val_size: Union[None, int, float] = None,
                  extrapolation: Extrapolation = None,
                  random_state: int = 0,
                  shuffle: bool = True,
                  stratify: Optional[Vector] = None) -> Dict:
    """Splits the input data.

    :param dataset:
        The input data vectors.

    :param test_size:
        The percentage of data left for testing (ignored in case of extrapolation).

    :param val_size:
        The percentage of data left for validation. If zero, no validation set is returned.

    :param extrapolation:
        Whether to split the data randomly or to test on a given percentage of extrapolated data.

    :param random_state:
        Controls the seed for the shuffling applied to the data before applying the split.

    :param shuffle:
        Whether or not to shuffle the data before splitting (if shuffle=False then stratify must be None).

    :param stratify:
        If not None, data is split in a stratified fashion, using this as the class labels.

    :returns:
        A dictionary of datasets.
    """
    val_size = test_size if val_size is None else val_size
    val_size = val_size if isinstance(val_size, float) else val_size / len(dataset[0])
    test_size = test_size if isinstance(test_size, float) else test_size / len(dataset[0])
    # split train/test
    if extrapolation is None:
        splits = train_test_split(*dataset, test_size=test_size, random_state=random_state,
                                  shuffle=shuffle, stratify=stratify)
    else:
        x = dataset[0]
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
        for d in dataset:
            splits.append(d[train_mask])
            splits.append(d[test_mask])
    train_data, test_data = splits if len(dataset) == 1 else (splits[::2], splits[1::2])
    # split val/test only if necessary
    if val_size == 0.0:
        return {'train': train_data, 'test': test_data}
    else:
        splits = train_test_split(*train_data, test_size=val_size, random_state=random_state,
                                  shuffle=shuffle, stratify=stratify)
        train_data, val_data = splits if len(dataset) == 1 else (splits[::2], splits[1::2])
        return {'train': train_data, 'validation': val_data, 'test': test_data}


def cross_validate(*dataset: ValidationArgs,
                   num_folds: int = 10,
                   random_state: int = 0,
                   shuffle: bool = True,
                   stratify: Optional[Vector] = None) -> List[Dict]:
    """Splits the input data in folds.

    :param dataset:
        The input data vectors.

    :param num_folds:
        The number of folds.

    :param random_state:
        Controls the seed for the shuffling applied to the data before applying the split.

    :param shuffle:
        Whether or not to shuffle the data before splitting (if shuffle=False then stratify must be None).

    :param stratify:
        Either None (no stratification) or the vector of labels.

    :returns:
        A list of dictionaries of datasets.
    """
    kf = KFold if stratify is None else StratifiedKFold
    kf = kf(n_splits=num_folds, random_state=random_state, shuffle=shuffle)
    folds = []
    for tr, vl in kf.split(X=dataset[0], y=stratify):
        folds.append(
            {'train': tuple([v.iloc[tr] for v in dataset]), 'validation': tuple([v.iloc[vl] for v in dataset])})
    return folds
