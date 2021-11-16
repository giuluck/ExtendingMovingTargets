"""Basic Logger Interface."""

from typing import Dict

from moving_targets.callbacks.callback import Callback


class Logger(Callback):
    """Basic interface for a Moving Targets Logger callback."""

    def __init__(self):
        """"""
        super(Logger, self).__init__()

        self._cache: Dict = {}
        """The internally stored cache."""

    def log(self, **cache):
        """Adds the given keyword argument to the inner cache.

        :param cache:
            Key-value pairs to be added to the cache (if the key is already present, the value is overwrote).
        """
        self._cache.update(cache)
