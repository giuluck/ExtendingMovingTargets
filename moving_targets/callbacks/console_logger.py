import time
from typing import Dict, Tuple, Any, Optional

from moving_targets.callbacks import Callback


class ConsoleLogger(Callback):
    def __init__(self):
        super(ConsoleLogger, self).__init__()
        self.time: Optional[float] = None

    def on_iteration_start(self, macs, x, y, val_data: Dict[str, Tuple[Any, Any]], iteration: int, **kwargs):
        print(f'-------------------- ITERATION: {iteration:02} --------------------')
        self.time = time.time()

    def on_iteration_end(self, macs, x, y, val_data: Dict[str, Tuple[Any, Any]], iteration: int, **kwargs):
        print(f'Time: {time.time() - self.time:.4f} s')
        self.time = None

    def on_process_end(self, macs, val_data: Dict[str, Tuple[Any, Any]], **kwargs):
        print('-------------------------------------------------------')
