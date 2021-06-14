"""Processing utils."""

from typing import Union, Dict, Tuple, List, Optional as Opt, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from moving_targets.util.typing import Matrix, Vector
from src.util.typing import Methods, Extrapolation


class Scaler:
    """Custom scaler that is able to treat each feature separately.

    Args:
        methods: either a string/tuple, a list of strings/tuples, or a dictionary of strings/tuples.
                 Each string/tuple represents a method:
                 - 'std', or 'standardize', standardizes the feature
                 - 'norm', or 'normalize', or 'minmax', normalizes the feature from (min, max) into (0, 1)
                 - 'zero', or 'max', or 'zeromax', normalizes the feature from (0, max) into (0, 1)
                 - Tuple[Number, Number], normalizes the feature from (t1, t2) into (0, 1)
                 - None, performs no scaling
                 If a single value is passed, all the features are scaled with the same method.
                 If a list or a dictionary is passed, each feature is scaled with the respective method.
    """

    def __init__(self, methods: Methods = 'std'):
        super(Scaler, self).__init__()
        self.methods = methods
        self.translation: Opt[np.ndarray] = None
        self.scaling: Opt[np.ndarray] = None

    def fit(self, data: Matrix):
        """Fits the scaler parameters.

        Args:
            data: the matrix/dataframe of samples.

        Returns:
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

    def transform(self, data: Matrix) -> Matrix:
        """Transforms the data according to the scaler parameters.

        Args:
            data: the matrix/dataframe of samples.

        Returns:
            The scaled data.
        """
        return (data - self.translation) / self.scaling

    def fit_transform(self, data: Matrix) -> Matrix:
        """Fits the scaler parameters.

        Args:
            data: the matrix/dataframe of samples.

        Returns:
            The scaled data.
        """
        return self.fit(data).transform(data)

    def inverse_transform(self, data: Matrix) -> Matrix:
        """Inverts the scaling according to the scaler parameters.

        Args:
            data: the previously scaled matrix/dataframe of samples.

        Returns:
            The original data.
        """
        return (data * self.scaling) + self.translation

    @staticmethod
    def get_default(num_features: int) -> Any:
        """Builds a blank scaler which works on data with the given number of features.

        Args:
            num_features: the number of features.

        Returns:
            A blank scaler.
        """
        return Scaler(methods=None).fit(data=[[0.] * num_features])


Scalers = Union[Opt[Scaler], Tuple[Opt[Scaler], Opt[Scaler]]]
SplitArgs = Union[Matrix, Vector]
ValidationArgs = Union[pd.DataFrame, pd.Series]


def split_dataset(*args: SplitArgs,
                  test_size: Union[int, float] = 0.2,
                  val_size: Union[None, int, float] = None,
                  extrapolation: Extrapolation = None,
                  **kwargs) -> Dict:
    """Splits the input data.

    Args:
        *args: the input data vectors.
        test_size: the percentage of data left for testing (ignored in case of extrapolation).
        val_size: the percentage of data left for validation. If zero, no validation set is returned.
        extrapolation: whether to split the data randomly or to test on a given percentage of extrapolated data.
        **kwargs: 'sklearn.model_selection.train_test_split' arguments.

    Returns:
        A dictionary of datasets.
    """
    # handle default values
    split_args = dict(shuffle=True, random_state=0)
    split_args.update(kwargs)
    val_size = test_size if val_size is None else val_size
    val_size = val_size if isinstance(val_size, float) else val_size / len(args[0])
    test_size = test_size if isinstance(test_size, float) else test_size / len(args[0])
    # split train/test
    if extrapolation is None:
        splits = train_test_split(*args, test_size=test_size, **split_args)
    else:
        x = args[0]
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
        for a in args:
            splits.append(a[train_mask])
            splits.append(a[test_mask])
    train_data, test_data = splits if len(args) == 1 else (splits[::2], splits[1::2])
    # split val/test only if necessary
    if val_size == 0.0:
        return {'train': train_data, 'test': test_data}
    else:
        splits = train_test_split(*train_data, test_size=val_size, **split_args)
        train_data, val_data = splits if len(args) == 1 else (splits[::2], splits[1::2])
        return {'train': train_data, 'validation': val_data, 'test': test_data}


def cross_validate(*args: ValidationArgs, num_folds: int = 10, stratify: Opt[Vector] = None, **kwargs) -> List[Dict]:
    """Splits the input data in folds.

    Args:
        *args: the input data vectors.
        num_folds: the number of folds.
        stratify: either None (no stratification) or the vector of labels.
        **kwargs: 'sklearn.model_selection.KFold' or 'sklearn.model_selection.StratifiedKFold' arguments.

    Returns:
        A list of dictionaries of datasets.
    """
    split_args = dict(shuffle=True, random_state=0)
    split_args.update(kwargs)
    kf = KFold if stratify is None else StratifiedKFold
    kf = kf(n_splits=num_folds, **split_args)
    folds = []
    for tr, vl in kf.split(X=args[0], y=stratify):
        folds.append({'train': tuple([v.iloc[tr] for v in args]), 'validation': tuple([v.iloc[vl] for v in args])})
    return folds
