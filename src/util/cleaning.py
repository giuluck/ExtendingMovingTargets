"""Data Cleaning utils."""

from typing import Union, List, Dict, Optional as Opt

import numpy as np
import pandas as pd


class FeatureInfo:
    """Utility class to manage the features of the dataset during preprocessing.

    Args:
        kind: the column dtype.
        alias: the column alias.
    """

    def __init__(self, kind: Opt[str] = None, alias: Opt[str] = None):
        self.kind: Opt[str] = kind
        self.alias: Opt[str] = alias


def clean_dataframe(df: pd.DataFrame, info: Dict[str, FeatureInfo]) -> pd.DataFrame:
    """Cleans the dataframe following the metadata given in the dictionary of FeatureInfo.

    Args:
        df: the input dataframe.
        info: the dictionary of metadata associated to each feature (the key of the dictionary).

    Returns:
        The cleansed dataset, with columns ordered as in info.
    """
    df = df[[k for k in info.keys()]]
    df = df.astype({k: v.kind for k, v in info.items() if v.kind is not None})
    df = df.rename(columns={k: v.alias for k, v in info.items() if v.alias is not None})
    return df
