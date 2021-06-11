"""Type aliases."""

from typing import Union, List, Tuple, Dict

import numpy as np
import pandas as pd

Number = Union[int, float]
Vector = Union[pd.Series, np.ndarray, List[Number]]
Matrix = Union[pd.DataFrame, np.ndarray, List[List[Number]]]
Data = Tuple[Matrix, Vector]
Dataset = Dict[str, Data]
Splits = Union[Dataset, List[Dataset]]
Iteration = Union[int, str]
Classes = Union[int, Vector]
MonotonicitiesList = List[Tuple[int, int]]
MonotonicitiesMatrix = Union[np.int32, np.ndarray]
