"""Console Logger Callback."""

import time
from typing import Optional as Opt

from moving_targets.callbacks.callback import Callback
from moving_targets.util.typing import Matrix, Vector, Dataset, Iteration


class ConsoleLogger(Callback):
    """Callback which logs basic information on screen during the MACS training."""

    def __init__(self):
        super(ConsoleLogger, self).__init__()
        self.time: Opt[float] = None

    # noinspection PyMissingOrEmptyDocstring
    def on_iteration_start(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        print(f'-------------------- ITERATION: {iteration:02} --------------------')
        self.time = time.time()

    # noinspection PyMissingOrEmptyDocstring
    def on_iteration_end(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        print(f'Time: {time.time() - self.time:.4f} s')
        self.time = None

    # noinspection PyMissingOrEmptyDocstring
    def on_process_end(self, macs, val_data: Opt[Dataset], **kwargs):
        print('-------------------------------------------------------')
