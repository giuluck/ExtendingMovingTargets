"""Console Logger Callback."""

import time
from typing import Optional

from moving_targets.callbacks.callback import Callback
from moving_targets.util.typing import Iteration, Dataset


class ConsoleLogger(Callback):
    """Callback which logs basic information on screen during the `MACS` training."""

    def __init__(self):
        """"""
        super(ConsoleLogger, self).__init__()

        self._time: Optional[float] = None
        """An internal variable used to compute elapsed time between routines."""

    def on_iteration_start(self, macs, x, y, val_data: Optional[Dataset], iteration: Iteration, **additional_kwargs):
        # Prints the current iteration and stores the initial time.
        print(f'-------------------- ITERATION: {iteration:02} --------------------')
        self._time = time.time()

    def on_iteration_end(self, macs, x, y, val_data: Optional[Dataset], iteration: Iteration, **additional_kwargs):
        # Prints the time elapsed from the iteration start.
        print(f'Time: {time.time() - self._time:.4f} s')
        self._time = None

    def on_process_end(self, macs, val_data: Optional[Dataset], **additional_kwargs):
        # Prints a final separator.
        print('-------------------------------------------------------')
