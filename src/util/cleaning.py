"""Data Cleaning utils."""

from typing import Dict, Optional

import pandas as pd


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
