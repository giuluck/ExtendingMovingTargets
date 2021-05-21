"""Type aliases."""

import numpy as np
from typing import Union, List, Dict, Callable, Tuple, Optional, Any

from moving_targets.util.typing import Matrix, Number
from src.util.preprocessing import Scaler

Augmented = Union[int, List[int], Dict[int]]
AugmentedData = Tuple[Matrix, Matrix]
SamplingFunctions = Dict[str, Tuple[int, Callable]]
Directions = Union[int, List[int], np.ndarray]
Method = Union[None, str, Tuple[Number, Number]]
Methods = Union[Method, List[Method], Dict[str, Method]]
Scalers = Union[None, Scaler, Tuple[Scaler, Scaler]]
Figsize = Optional[Tuple[int, int]]
TightLayout = Optional[bool]
Rng = Any
Extrapolation = Union[None, bool, float, List[float], Dict[str, float]]
