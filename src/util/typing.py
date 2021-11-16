"""Type aliases."""

from typing import Union, List, Dict, Callable, Tuple, Optional, Any

from moving_targets.util.typing import Matrix, Number

Augmented = Union[int, Tuple[int], List[int], Dict[str, int]]
"""The number of augmented samples.

It can be a single integer number, a tuple/list of integers, or a dictionary of
integers indexed via the name of the feature to augment.
"""

AugmentedData = Tuple[Matrix, Matrix]
"""A tuple of matrices of kind (<x augmented>, <y augmented>)."""

SamplingFunctions = Dict[str, Tuple[int, Callable]]
"""A dictionary that assigns to each feature name (str) a tuple (<n>, <fn>), where <n> is the number of augmented
samples and <fn> is a callable function used for sampling."""

Method = Union[None, str, Tuple[Number, Number]]
"""Scaling method.

It can be either None (no scaling), a string representing the kind of scaling, or a tuple of integers representing the
lower and upper bounds, respectively.
"""

Methods = Union[Method, List[Method], Dict[str, Method]]
"""Multiple scaling methods.

It can be either a single `Method`, a list of them, or a dictionary where each method is indexed via the feature name.
"""

Figsize = Optional[Tuple[int, int]]
"""The figsize used in `matplotlib.pyplot.show()` method."""

TightLayout = Optional[bool]
"""The tight_layout used in `matplotlib.pyplot.show()` method."""

Rng = Any
"""Random Number Generator type."""

Extrapolation = Union[None, bool, float, List[float], Dict[str, float]]
"""The extrapolation type used when splitting the dataset.

It can be either None (no extrapolation), a boolean value, a single float or a list of them (one for each feature,
ordered), or a dictionary of float values indexed via the feature name.
"""
