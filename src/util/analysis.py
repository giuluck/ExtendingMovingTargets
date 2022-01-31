from typing import Optional, Callable, List, Dict

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb, to_hex


def cartesian_product(fixed_parameters: Optional[Dict] = None, **parameters: List) -> List[Dict]:
    """Creates a combinatorial list of configurations given a dictionary of fixed parameters and a series of keyword
       arguments representing a list of variable parameters."""
    fixed_parameters = {} if fixed_parameters is None else fixed_parameters
    if len(parameters) == 0:
        return [fixed_parameters]
    else:
        cart_product = []
        parameter, values = parameters.popitem()
        for value in values:
            new_parameters = {**fixed_parameters, parameter: value}
            sub_product = cartesian_product(fixed_parameters=new_parameters, **parameters)
            cart_product.append(sub_product)
        return [parameters for sub_product in cart_product for parameters in sub_product]


def set_pandas_options(min_rows: Optional[int] = 1000,
                       max_rows: Optional[int] = 1000,
                       max_columns: Optional[int] = 1000,
                       max_colwidth: Optional[int] = 1000,
                       width: Optional[int] = 100000,
                       precision: Optional[int] = None,
                       float_format: Optional[Callable] = '{:.2f}'.format):
    """Sets a range of pandas options."""
    for key, value in locals().items():
        if value is not None:
            pd.set_option(f'display.{key}', value)


class ColorFader:
    """Interpolates numerical values on a hypercube having certain colours at the corners.

    - 'colours' are the colours to be placed at the corner of the hypercube. Their length must be a power of two.
    - 'bounds' is the vector of min/max values for each dimension. If None, min=0 and max=1 for each dimension.
    - 'error' specifies how to deal with values that are outside the bounds, it can be:
        > 'raise', which raises a `ValueError` if there is at least one value outside the bounds.
        > 'input', which clips the input values if they are outside the bounds.
        > 'output', which clips the output values if they are outside the bounds.
    """

    def __init__(self, *colours, bounds: Optional = None, error: str = 'raise'):
        assert np.log2(len(colours)) % 1 == 0, "Please provide a number of colors which is a power of two"
        assert error in ['raise', 'input', 'output'], f"'{error}' is not a valid error kind"

        self._dim: int = int(np.log2(len(colours)))
        self._colours: np.ndarray = np.array([to_rgb(c) for c in colours])
        self._error: str = error
        self._translation: np.ndarray = np.zeros_like(self._dim)
        self._scale: np.ndarray = np.ones_like(self._dim)

        if bounds is not None:
            bounds = np.array(bounds).reshape(2, self._dim)
            self._translation = bounds[0]
            self._scale = bounds[1] - bounds[0]

    def __call__(self, *vector):
        """Interpolates the given vector of values to be interpolated wrt to the defined hypercube"""
        # handle user input
        assert len(vector) == self._dim, f'Please provide an input vector having size {len(vector)}'
        values = (np.array(vector) - self._translation) / self._scale
        values_clipped = values.clip(min=0, max=1)
        # if not values and values_clipped are not close, it means that some values were out of bounds
        if not np.allclose(values, values_clipped) and self._error == 'raise':
            raise ValueError(f'Value {vector} is out of the minmax scale')
        # if error is set on 'input', use clipped values
        elif self._error == 'input':
            values = values_clipped
        # color interpolation
        output = np.zeros(3)
        for corner, color in enumerate(self._colours):
            corner = np.array([int(c) for c in np.binary_repr(corner, self._dim)][::-1])
            distance = np.max(np.abs(values - corner))
            output += (1 - distance) * color
        # with values within the limits (or clipped values), this next clipping is useless
        # however, in case the error is set on 'output', some values may still be out of bounds
        return to_hex(output.clip(min=0, max=1))
