"""Type aliases."""

from typing import Union, List, Dict, Tuple

Method = Union[None, str, Tuple[float, float]]
"""Scaling method. It can be either None (no scaling), a string representing the kind of scaling, or a tuple of
integers representing the lower and upper bounds, respectively.
"""

Extrapolation = Union[None, bool, float, List[float], Dict[str, float]]
"""The extrapolation type used when splitting the dataset. It can be either None (no extrapolation), a boolean value,
a single float or a list of them (one for each feature, ordered), or a dictionary of float values indexed via the
feature name.
"""
