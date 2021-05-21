import time
from typing import List, Optional, Tuple, Any, Dict, Callable

from moving_targets.callbacks import Logger, FileLogger, History, ConsoleLogger, Callback
from moving_targets.learners import Learner
from moving_targets.masters import Master
from moving_targets.metrics import Metric


class MACS(Logger):
    def __init__(self, learner: Learner, master: Master, init_step: str = 'pretraining',
                 metrics: Optional[List[Metric]] = None):
        super(MACS, self).__init__()
        assert init_step in ['pretraining', 'projection'], "initial step should be 'pretraining' or 'projection'"
        self.learner: Learner = learner
        self.master: Master = master
        self.init_step: str = init_step
        self.metrics: List[Metric] = [] if metrics is None else metrics
        self.history: History = History()
        self.fitted: bool = False
        self.time: Optional[float] = None

    def fit(self, x, y, iterations: int = 1, val_data: Dict[str, Tuple[Any, Any]] = None,
            callbacks: Optional[List[Callback]] = None, verbose: Any = 2):
        # check user input
        assert iterations >= 0, 'the number of iterations should be non-negative'
        assert iterations > 0 or self.init_step == 'pretraining', 'if projection, iterations should be a positive value'
        assert isinstance(verbose, bool) or (isinstance(verbose, int) and verbose in [0, 1, 2]), 'unknown verbosity'
        val_data = {} if val_data is None else (val_data if isinstance(val_data, dict) else {'val': val_data})

        # handle callbacks and verbosity
        callbacks = [] if callbacks is None else callbacks
        print_callback = []
        if isinstance(verbose, int):
            print_callback = [FileLogger()] if verbose == 2 else ([ConsoleLogger()] if verbose == 1 else [])
        elif isinstance(verbose, bool):
            print_callback = [FileLogger()] if verbose else []
        callbacks = print_callback + callbacks
        self._update_callbacks(callbacks, lambda c: c.on_process_start(self, x=x, y=y, val_data=val_data))

        # handle pretraining
        kwargs = dict(x=x, y=y, val_data=val_data, iteration=0)
        if self.init_step == 'pretraining':
            self._update_callbacks(callbacks, lambda c: c.on_pretraining_start(self, **kwargs))
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self.learner.fit(self, x, y, iteration='pretraining')
            self.fitted = True
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self._update_callbacks(callbacks, lambda c: c.on_pretraining_end(self, **kwargs))

        # algorithm core
        for iteration in range(1, iterations + 1):
            kwargs['iteration'] = iteration
            self._update_callbacks(callbacks, lambda c: c.on_iteration_start(self, **kwargs))
            self._update_callbacks(callbacks, lambda c: c.on_adjustment_start(self, **kwargs))
            # ---------------------------------------------- MASTER  STEP ----------------------------------------------
            yj, kw = self.master.adjust_targets(self, x=kwargs['x'], y=kwargs['y'], iteration=iteration), {}
            if isinstance(yj, tuple):
                yj, kw = yj
                kwargs.update(kw)
            # ---------------------------------------------- MASTER  STEP ----------------------------------------------
            self._update_callbacks(callbacks, lambda c: c.on_adjustment_end(self, adjusted_y=yj, **kwargs))
            self._update_callbacks(callbacks, lambda c: c.on_training_start(self, **kwargs))
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self.learner.fit(self, y=yj, **{k: v for k, v in kwargs.items() if k not in ['val_data', 'y']})
            self.fitted = True
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self._update_callbacks(callbacks, lambda c: c.on_training_end(self, **kwargs))
            self._update_callbacks(callbacks, lambda c: c.on_iteration_end(self, **kwargs))
        self._update_callbacks(callbacks, lambda c: c.on_process_end(self, val_data=val_data))
        return self.history

    def predict(self, x):
        assert self.fitted, 'The model has not been fitted yet, please call method .fit()'
        return self.learner.predict(x)

    def evaluate(self, x, y):
        p = self.predict(x)
        return {metric.__name__: metric(x, y, p) for metric in self.metrics}

    def on_iteration_start(self, macs, x, y, val_data: Dict[str, Tuple[Any, Any]], iteration: Any, **kwargs):
        self.time = time.time()

    def on_iteration_end(self, macs, x, y, val_data: Dict[str, Tuple[Any, Any]], iteration: Any, **kwargs):
        logs = {'iteration': iteration, 'elapsed time': time.time() - self.time}
        # log metrics on training data
        p = self.predict(x)
        for metric in self.metrics:
            logs[metric.__name__] = metric(x, y, p)
        # log metrics on validation data
        for name, (xx, yy) in val_data.items():
            pp = self.predict(xx)
            for metric in self.metrics:
                logs[f'{name}_{metric.__name__}'] = metric(xx, yy, pp)
        self.log(**logs)

    def _update_callbacks(self, callbacks: List[Callback], routine: Callable):
        # run callback routine for macs object itself and history object
        routine(self)
        # run callback routine for the history logger and for each external callback
        for callback in callbacks + [self.history]:
            # if the callback is a logger, log the internal cache before calling the routine (and eventually clear it)
            if isinstance(callback, Logger):
                callback.log(**self.cache)
            routine(callback)
        self.cache = {}
