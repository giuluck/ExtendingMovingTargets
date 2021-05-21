"""Type aliases."""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Dict

Number = Union[int, float]
Vector = Union[pd.Series, np.ndarray, List[Number]]
Matrix = Union[pd.DataFrame, np.ndarray, List[List[Number]]]
Data = Tuple[Matrix, Vector]
Dataset = Dict[str, Data]
Iteration = Union[int, str]
Classes = Union[int, Vector]
Monotonicities = List[Tuple[int, int]]
