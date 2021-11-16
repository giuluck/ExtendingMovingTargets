"""Type aliases."""

from typing import Union, List, Tuple, Dict

import numpy as np
import pandas as pd

Number = Union[int, float]
"""A scalar."""

Vector = Union[pd.Series, np.ndarray, List[Number]]
"""A vector of real numbers."""

Matrix = Union[pd.DataFrame, np.ndarray, List[List[Number]]]
"""A matrix of real numbers."""

Data = Tuple[Matrix, Vector]
"""A tuple (x, y), where x is a `Matrix` and y is a `Vector`."""

Dataset = Dict[str, Data]
"""A dictionary where the key is string representing the name of the split, and the value is a `Data` instance."""

Splits = Union[Dataset, List[Dataset]]
"""Either a single `Dataset` or a list of them."""

Iteration = Union[int, str]
"""A Moving Targets iteration, either an integer value or a string."""

Classes = Union[int, Vector]
"""Either a single integer representing the number of classes, or a vector representing the actual classes."""

MonotonicitiesList = List[Tuple[int, int]]
"""A list of tuples (<higher>, <lower>), where the values represent the indices of the greater and the lower samples."""

MonotonicitiesMatrix = Union[np.int32, np.ndarray]
"""Either a numpy scalar, vector or matrix which contains {-1, 0, 1} depending on the expected monotonicity."""
