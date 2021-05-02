import time

from moving_targets.callbacks import Callback


class ConsoleLogger(Callback):
    def __init__(self):
        super(ConsoleLogger, self).__init__()
        self.time = None

    def on_iteration_start(self, macs, x, y, val_data, iteration, **kwargs):
        print(f'-------------------- ITERATION: {iteration:02} --------------------')
        self.time = time.time()

    def on_iteration_end(self, macs, x, y, val_data, iteration, **kwargs):
        print(f'Time: {time.time() - self.time:.4f} s')
        self.time = None
