"""File Logger Callback"""

import sys
from typing import List, Set, Optional as Opt

from moving_targets.callbacks.logger import Logger
from moving_targets.util.typing import Matrix, Vector, Dataset, Iteration

SEPARATOR: str = '--------------------------------------------------'


class FileLogger(Logger):
    """Logs the training data and information on a specified filepath or on the standard output.

    Args:
        filepath: path string in which to put the log. If None, writes in the standard output.
        routines: list of routine names after which log the cached data. If None, logs after each routine.
        log_empty: whether or not to log the routine name if there is no cached data.
        sort_keys: whether or not to sort the cache data by key before logging.
        separator: string separator written between each routine.
        end: line end.
    """

    def __init__(self, filepath: Opt[str] = None, routines: Opt[List[str]] = None, log_empty: bool = False,
                 sort_keys: bool = False, separator: str = SEPARATOR, end: str = '\n'):
        super(FileLogger, self).__init__()
        self.filepath: str = filepath
        self.routines: Opt[Set[str]] = None if routines is None else set(routines)
        self.log_empty: bool = log_empty
        self.sort_keys: bool = sort_keys
        self.separator: str = separator
        self.end: str = end
        self.logged_once: bool = False
        # reset file content
        if self.filepath is not None:
            open(filepath, 'w').close()

    # noinspection PyMissingOrEmptyDocstring
    def on_process_start(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], **kwargs):
        self._write_on_file('START PROCESS', 'on_process_start')

    # noinspection PyMissingOrEmptyDocstring
    def on_process_end(self, macs, val_data: Opt[Dataset], **kwargs):
        self._write_on_file('END PROCESS', 'on_process_end')

    # noinspection PyMissingOrEmptyDocstring
    def on_pretraining_start(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], **kwargs):
        self._write_on_file('START PRETRAINING', 'on_pretraining_start')

    # noinspection PyMissingOrEmptyDocstring
    def on_pretraining_end(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], **kwargs):
        self._write_on_file('END PRETRAINING', 'on_pretraining_end')

    # noinspection PyMissingOrEmptyDocstring
    def on_iteration_start(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        self._write_on_file(f'START ITERATION', 'on_iteration_start')

    # noinspection PyMissingOrEmptyDocstring
    def on_iteration_end(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        self._write_on_file(f'END ITERATION', 'on_iteration_end')

    # noinspection PyMissingOrEmptyDocstring
    def on_training_start(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        self._write_on_file('START TRAINING', 'on_training_start')

    # noinspection PyMissingOrEmptyDocstring
    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        self._write_on_file('END TRAINING', 'on_training_end')

    # noinspection PyMissingOrEmptyDocstring
    def on_adjustment_start(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        self._write_on_file('START ADJUSTMENT', 'on_adjustment_start')

    # noinspection PyMissingOrEmptyDocstring
    def on_adjustment_end(self, macs, x: Matrix, y: Vector, adjusted_y: Vector, val_data: Opt[Dataset],
                          iteration: Iteration, **kwargs):
        self._write_on_file('END ADJUSTMENT', 'on_adjustment_end')

    def _write_on_file(self, message: str, routine_name: str):
        if (self.routines is None or routine_name in self.routines) and (self.log_empty or len(self.cache) > 0):
            # open file
            file = sys.stdout if self.filepath is None else open(self.filepath, 'a', encoding='utf8')
            # write initial separator if needed
            if not self.logged_once:
                file.write(f'{self.separator}{self.end}')
                self.logged_once = True
            # write message and cached items if present
            file.write(f'{message}{self.end}')
            cache = {k: self.cache[k] for k in sorted(self.cache)} if self.sort_keys else self.cache
            for k, v in cache.items():
                file.write(f'> {str(k)} = {str(v)}{self.end}')
            # write write separator and empty cache
            file.write(f'{self.separator}{self.end}')
            self.cache = {}
            # close file if not stdout
            if self.filepath is not None:
                file.close()
