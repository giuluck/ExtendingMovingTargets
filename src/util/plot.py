"""Plot utils."""

from typing import Optional

import numpy as np
from matplotlib.colors import to_rgb, to_hex

from moving_targets.util.typing import Vector


class ColorFader:
    """Interpolates numerical values on a hypercube having certain colours at the corners.

    Args:
        *args: the colours to be placed at the corner of the hypercube. This length must be a power of two.
        bounds: the vector of min/max values for each dimension. If None, min=0 and max=1 for each dimension.
        error: how to deal with values that are outside the bounds:
               - 'raise', which raises a `ValueError` if there is at least one value outside the bounds.
               - 'input', which clips the input values if they are outside the bounds.
               - 'output', which clips the output values if they are outside the bounds.

    Raises:
        `AssertionError` if the passed number of colours is not a power of two.
        `AssertionError` if 'error' is not in ['raise', 'input', 'output'].
    """

    def __init__(self, *args, bounds: Optional[Vector] = None, error: str = 'raise'):
        assert np.log2(len(args)) % 1 == 0, "Please provide a number of colors which is a power of two"
        assert error in ['raise', 'input', 'output'], "'error' parameter should be in ['raise', 'input', 'output']"
        self.dim = int(np.log2(len(args)))
        self.colors = np.array([to_rgb(c) for c in args])
        self.translation = np.zeros_like(self.dim)
        self.scale = np.ones_like(self.dim)
        if bounds is not None:
            bounds = np.array(bounds).reshape(2, self.dim)
            self.translation = bounds[0]
            self.scale = bounds[1] - bounds[0]
        self.error = error

    def __call__(self, *args):
        """Interpolates the given vector wrt to the defined hypercube.

        Args:
            *args: The vector of values to be interpolated.

        Returns:
            An RGB colour.

        Raises:
            `AssertionError` if the passed number of values has not the correct dimensions, i.e., log2(corners).
            `ValueError` if a value is out of bounds and error is set to 'raise'.
        """
        # handle user input
        assert len(args) == self.dim, f'Please provide an input vector having size {len(args)}'
        values = (np.array(args) - self.translation) / self.scale
        values_clipped = values.clip(min=0, max=1)
        # if not values and values_clipped are not close, it means that some values were out of bounds
        if not np.allclose(values, values_clipped) and self.error == 'raise':
            raise ValueError(f'Value {args} is out of the minmax scale')
        # if error is set on 'input', use clipped values
        elif self.error == 'input':
            values = values_clipped
        # color interpolation
        output = np.zeros(3)
        for corner, color in enumerate(self.colors):
            corner = np.array([int(c) for c in np.binary_repr(corner, self.dim)][::-1])
            distance = np.max(np.abs(values - corner))
            output += (1 - distance) * color
        # with values within the limits (or clipped values), this next clipping is useless
        # however, in case the error is set on 'output', some values may still be out of bounds
        return to_hex(output.clip(min=0, max=1))
