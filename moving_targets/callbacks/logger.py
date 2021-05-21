"""Basic Logger Interface."""

from typing import Dict

from moving_targets.callbacks.callback import Callback


class Logger(Callback):
    """Basic interface for a Moving Target's logger callback."""

    def __init__(self):
        super(Logger, self).__init__()
        self.cache: Dict = {}

    def log(self, **kwargs):
        """Adds the given keyword argument to the inner cache.

        Args:
            **kwargs: optional keyword arguments.
        """
        self.cache.update(kwargs)
