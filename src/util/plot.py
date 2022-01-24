"""Plot utils."""

from typing import Optional

import numpy as np
from matplotlib.colors import to_rgb, to_hex


class ColorFader:
    """Interpolates numerical values on a hypercube having certain colours at the corners."""

    def __init__(self, *colours, bounds: Optional = None, error: str = 'raise'):
        """
        :param colours:
            The colours to be placed at the corner of the hypercube. This length must be a power of two.

        :param bounds:
            The vector of min/max values for each dimension. If None, min=0 and max=1 for each dimension.

        :param error:
            How to deal with values that are outside the bounds:

            - 'raise', which raises a `ValueError` if there is at least one value outside the bounds.
            - 'input', which clips the input values if they are outside the bounds.
            - 'output', which clips the output values if they are outside the bounds.

        :raise `AssertionError`:
            If the passed number of colours is not a power of two.

        :raise `AssertionError`:
            If 'error' is not in ['raise', 'input', 'output'].
        """
        assert np.log2(len(colours)) % 1 == 0, "Please provide a number of colors which is a power of two"
        assert error in ['raise', 'input', 'output'], f"'{error}' is not a valid error kind"

        self._dim: int = int(np.log2(len(colours)))
        """The expected vector dimension, which is the base-2 logarithm of the size of corners (colours)."""

        self._colours: np.ndarray = np.array([to_rgb(c) for c in colours])
        """The array of corner colours."""

        self._error: str = error
        """The strategy to deal with values that are outside the bounds."""

        self._translation: np.ndarray = np.zeros_like(self._dim)
        """The inner translation vector."""

        self._scale: np.ndarray = np.ones_like(self._dim)
        """The inner scaling vector."""

        if bounds is not None:
            bounds = np.array(bounds).reshape(2, self._dim)
            self._translation = bounds[0]
            self._scale = bounds[1] - bounds[0]

    def __call__(self, *vector):
        """Interpolates the given vector wrt to the defined hypercube.

        :param vector:
            The vector of values to be interpolated.

        :return:
            An RGB colour.

        :raise `AssertionError`:
            If the passed number of values has not the correct dimensions, i.e., log2(corners).

        :raise `ValueError`:
            If a value is out of bounds and error is set to 'raise'.
        """
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
