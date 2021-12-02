"""Type aliases."""

from typing import Union, List, Tuple, Dict, Any

Number = Union[int, float]
"""A scalar."""

Data = Tuple[Any, Any]
"""A tuple (x, y), where x is the input and y is the output data."""

Dataset = Dict[str, Data]
"""A dictionary where the key is string representing the name of the split, and the value is a `Data` instance."""

Iteration = Union[int, str]
"""A Moving Targets iteration, either an integer value or a string."""

Solution = Union[Any, Tuple[Any, Dict]]
"""A Moving Targets solution, either the single output or the output paired with a dictionary of additional info."""

Classes = Union[int, List]
"""Either a single integer representing the number of classes, or a list representing the actual classes."""

MonotonicitiesList = List[Tuple[int, int]]
"""A list of tuples (<higher>, <lower>), where the values represent the indices of the greater and the lower samples."""
