from typing import Any, Optional

import numpy as np
from matplotlib.colors import to_rgb, to_hex


class ColorFader:
    def __init__(self, *args, bounds: Optional[Any] = None, error: str = 'raise'):
        assert np.log2(len(args)) % 1 == 0, "Please provide a number of colors which is a power of two"
        assert error in ['raise', 'input', 'output'], "'error' parameter should be either 'raise', 'input', or 'output'"
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
        # handle user input
        assert len(args) == self.dim, f'Please provide an input vector having size {len(args)}'
        values = (np.array(args) - self.translation) / self.scale
        values_clipped = values.clip(min=0, max=1)
        if not np.allclose(values, values_clipped) and self.error == 'raise':
            raise ValueError(f'Value {args} is out of the minmax scale')
        elif self.error == 'input':
            values = values_clipped
        # color interpolation
        output = np.zeros(3)
        for corner, color in enumerate(self.colors):
            corner = np.array([int(c) for c in np.binary_repr(corner, self.dim)][::-1])
            distance = np.max(np.abs(values - corner))
            output += (1 - distance) * color
        return to_hex(output.clip(min=0, max=1))
