import sys

from moving_targets.callbacks.logger import Logger

SEPARATOR = '--------------------------------------------------'


class FileLogger(Logger):
    def __init__(self, filepath=None, routines=None, log_empty=False, sort_keys=False, separator=SEPARATOR, end='\n'):
        super(FileLogger, self).__init__()
        self.filepath = filepath
        self.routines = None if routines is None else set(routines)
        self.log_empty = log_empty
        self.sort_keys = sort_keys
        self.separator = separator
        self.end = end
        self.logged_once = False
        # reset file content
        if self.filepath is not None:
            open(filepath, 'w').close()

    def on_process_start(self, macs, x, y, val_data, **kwargs):
        self._write_on_file('START PROCESS', 'on_process_start')

    def on_process_end(self, macs, val_data, **kwargs):
        self._write_on_file('END PROCESS', 'on_process_end')

    def on_pretraining_start(self, macs, x, y, val_data):
        self._write_on_file('START PRETRAINING', 'on_pretraining_start')

    def on_pretraining_end(self, macs, x, y, val_data, **kwargs):
        self._write_on_file('END PRETRAINING', 'on_pretraining_end')

    def on_iteration_start(self, macs, x, y, val_data, iteration, **kwargs):
        self._write_on_file(f'START ITERATION', 'on_iteration_start')

    def on_iteration_end(self, macs, x, y, val_data, iteration, **kwargs):
        self._write_on_file(f'END ITERATION', 'on_iteration_end')

    def on_training_start(self, macs, x, y, val_data, iteration):
        self._write_on_file('START TRAINING', 'on_training_start')

    def on_training_end(self, macs, x, y, val_data, iteration, **kwargs):
        self._write_on_file('END TRAINING', 'on_training_end')

    def on_adjustment_start(self, macs, x, y, val_data, iteration, **kwargs):
        self._write_on_file('START ADJUSTMENT', 'on_adjustment_start')

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        self._write_on_file('END ADJUSTMENT', 'on_adjustment_end')

    def _write_on_file(self, message, routine_name):
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
