from src.moving_targets.callbacks import Callback


class Logger(Callback):
    def __init__(self):
        super(Logger, self).__init__()
        self.cache = {}

    def log(self, **kwargs):
        self.cache.update(kwargs)
