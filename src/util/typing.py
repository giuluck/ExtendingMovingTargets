"""Type aliases."""

from typing import Union, List, Dict, Callable, Tuple, Optional, Any

import numpy as np

from moving_targets.util.typing import Matrix, Number

Augmented = Union[int, Tuple[int], List[int], Dict[str, int]]
AugmentedData = Tuple[Matrix, Matrix]
SamplingFunctions = Dict[str, Tuple[int, Callable]]
Directions = Union[int, List[int], np.ndarray]
Method = Union[None, str, Tuple[Number, Number]]
Methods = Union[Method, List[Method], Dict[str, Method]]
Figsize = Optional[Tuple[int, int]]
TightLayout = Optional[bool]
Rng = Any
Extrapolation = Union[None, bool, float, List[float], Dict[str, float]]
