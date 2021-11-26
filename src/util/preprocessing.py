"""Processing utils."""

from typing import Union, Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

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
            - 'onehot', encodes categorical features as a vector
            - None, performs no scaling

            If a single value is passed, all the features are scaled with the same method.
            If a list or a dictionary is passed, each feature is scaled with the respective method.
        """
        super(Scaler, self).__init__()

        self.methods = methods
        """The scaling methods."""

        self._is_2d: bool = True
        """Whether or not the input data is 2d."""

        self._is_pandas: bool = True
        """Whether or not the input data is pandas-like."""

        self._onehot: Dict[int, Tuple[OneHotEncoder, object]] = {}
        """The dictionary of onehot encoders (paired with the previous column name, if any) indexed by column number."""

        self._translation: Optional[np.ndarray] = None
        """The translation vector."""

        self._scaling: Optional[np.ndarray] = None
        """The scaling vector."""

    @staticmethod
    def _handle_input(data: Union[Vector, Matrix]) -> Tuple[Matrix, Tuple[bool, bool]]:
        """Handles the input data and stores metadata on its dimension and data type.

        :param data:
            The input data.

        :return:
            A tuple containing the processed input and its metadata.
        """
        # handle non-dataframe data (convert to dataframe in case of series, to 2D array in case of non-pandas)
        is_2d, is_pandas = True, True
        if isinstance(data, pd.Series):
            is_2d = False
        elif not isinstance(data, pd.DataFrame):
            data = np.array(data)
            is_2d = data.ndim > 1
            is_pandas = False
            data = data if is_2d else data.reshape((-1, 1))
        return pd.DataFrame(data), (is_2d, is_pandas)

    @staticmethod
    def _handle_output(data: Matrix, is_2d: bool, is_pandas: bool) -> Union[Vector, Matrix]:
        """Handles the output data depending on its dimension and data type.

        :param data:
            The output data.

        :param is_2d:
            Whether the output must be returned in 2d or not (may differ from `self.is_2d` due to one hot encoding).

        :param is_pandas:
            Whether the output must be returned in pandas-like.

        :return:
            The processed output data.
        """
        # handle non-dataframe data (convert to dataframe in case of series, to 2D array in case of non-pandas)
        data = data if is_2d else data.iloc[:, 0]
        return data if is_pandas else data.values

    def fit(self, data: Union[Vector, Matrix]):
        """Fits the scaler parameters.

        :param data:
            The matrix/dataframe of samples.

        :return:
            The scaler itself.
        """
        # handle input
        data, (self._is_2d, self._is_pandas) = Scaler._handle_input(data=data)

        # handle methods
        if isinstance(self.methods, dict):
            methods = self.methods
        elif isinstance(self.methods, list):
            methods = {column: method for method, column in zip(self.methods, data.columns)}
        else:
            methods = {column: self.methods for column in data.columns}

        # handle one hot encoding
        df = data.copy()
        for index, (column, method) in enumerate(methods.items()):
            if method == 'onehot':
                encoder = OneHotEncoder()
                dummies = encoder.fit_transform(df[[column]])
                dummies = pd.DataFrame(dummies.todense(), index=data.index, columns=encoder.categories_[0], dtype=int)
                data = pd.concat([data.iloc[:, :index], dummies, data.iloc[:, index + 1:]], axis=1)
                self._onehot[index] = (encoder, df.columns[index])

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

    def transform(self, data: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
        """Transforms the data according to the scaler parameters.

        :param data:
            The matrix/dataframe of samples.

        :return:
            The scaled data.
        """
        # handle input
        data, _ = Scaler._handle_input(data=data)

        # handle one hot encoding and scale
        is_2d = self._is_2d
        for index, (encoder, _) in self._onehot.items():
            # put is_2d to True since, even though the data may have been 1d, now it is not anymore
            is_2d = True
            dummies = encoder.transform(data.iloc[:, index:index + 1])
            dummies = pd.DataFrame(dummies.todense(), index=data.index, columns=encoder.categories_[0], dtype=int)
            data = pd.concat([data.iloc[:, :index], dummies, data.iloc[:, index + 1:]], axis=1)
        data = (data - self._translation) / self._scaling

        # handle output
        return Scaler._handle_output(data=data, is_2d=is_2d, is_pandas=self._is_pandas)

    def fit_transform(self, data: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
        """Fits the scaler parameters.

        :param data:
            The matrix/dataframe of samples.

        :return:
            The scaled data.
        """
        return self.fit(data).transform(data)

    def inverse_transform(self, data: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
        """Inverts the scaling according to the scaler parameters.

        :param data:
            The previously scaled matrix/dataframe of samples.

        :return:
            The original data.
        """
        # handle input
        data, _ = Scaler._handle_input(data=data)

        # handle one hot encoding and scale
        data = (data * self._scaling) + self._translation
        for index, (encoder, column) in self._onehot.items():
            encoded = data.iloc[:, index:index + len(encoder.categories_[0])]
            decoded = encoder.inverse_transform(encoded)
            data = data.drop(columns=encoder.categories_[0])
            data.insert(loc=index, column=column, value=decoded)

        # handle output
        return Scaler._handle_output(data=data, is_2d=self._is_2d, is_pandas=self._is_pandas)

    @staticmethod
    def get_default(num_features: int) -> Any:
        """Builds a blank scaler which works on data with the given number of features.

        :param num_features:
            The number of features.

        :return:
            A blank scaler.
        """
        return Scaler(methods=None).fit(data=[[0.] * num_features])


Scalers = Union[Optional[Scaler], Tuple[Optional[Scaler], Optional[Scaler]]]
"""Either an (optional) scaler for both the input and the output data, or a tuple of (optional) scalers for the input
and the output data, respectively."""

SplitArgs = Union[Matrix, Vector]
"""Either a `Matrix` or a `Vector` that needs to be split into train/test."""

ValidationArgs = Union[pd.DataFrame, pd.Series]
"""Either a `DataFrame` or a `Series` that needs to be split for cross-validation."""


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

    :return:
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

    :return:
        A list of dictionaries of datasets.
    """
    kf = KFold if stratify is None else StratifiedKFold
    kf = kf(n_splits=num_folds, random_state=random_state, shuffle=shuffle)
    folds = []
    for tr, vl in kf.split(X=dataset[0], y=stratify):
        folds.append(
            {'train': tuple([v.iloc[tr] for v in dataset]), 'validation': tuple([v.iloc[vl] for v in dataset])})
    return folds
