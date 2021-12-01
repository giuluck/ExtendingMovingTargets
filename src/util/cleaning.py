"""Data Cleaning utils."""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler


class FeatureInfo:
    """Utility class to manage the features of the dataset during preprocessing."""

    def __init__(self, dtype: Optional[str] = None, alias: Optional[str] = None):
        """
        :param dtype:
            The column dtype.

        :param alias:
            The column alias.
        """

        self.dtype: Optional[str] = dtype
        """The column dtype."""

        self.alias: Optional[str] = alias
        """The column alias."""


def clean_dataframe(df: pd.DataFrame, info: Dict[str, FeatureInfo]) -> pd.DataFrame:
    """Cleans the dataframe following the metadata given in the dictionary of FeatureInfo.

    :param df:
        The input dataframe.

    :param info:
        The dictionary of metadata associated to each feature (the key of the dictionary).

    :return:
        The cleansed dataset, with columns ordered as in info.
    """
    df = df[[k for k in info.keys()]]
    df = df.astype({k: v.dtype for k, v in info.items() if v.dtype is not None})
    df = df.rename(columns={k: v.alias for k, v in info.items() if v.alias is not None})
    return df


def get_top_features(x, y, n: int = 10) -> list:
    """Get the n most relevant features of the input dataset. In order to do feature selection, we have to fix a model:
    here we use a random forest for simplicity, since feature importance is already implemented in sklearn routines.

    :param x:
        The input data.

    :param y:
        The output data.

    :param n:
        The number of features to be taken.

    :return:
        A list of n features ordered by importance.
    """
    x_scaled = MinMaxScaler().fit_transform(x)
    y_scaled = MinMaxScaler().fit_transform(y)
    features = list(x.columns)
    model = SelectKBest(k=n).fit(x_scaled, y_scaled.reshape((-1,)))
    # noinspection PyUnresolvedReferences
    feat_ranking = model.scores_
    ranked_features = np.take(features, np.argsort(feat_ranking))
    return list(ranked_features)[:n]
