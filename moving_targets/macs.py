"""Core of the Moving Target algorithm."""

import time
from typing import List, Dict, Callable, Union, Optional as Opt

from moving_targets.callbacks import Logger, FileLogger, History, ConsoleLogger, Callback
from moving_targets.learners import Learner
from moving_targets.masters import Master
from moving_targets.metrics import Metric
from moving_targets.util.typing import Matrix, Vector, Dataset, Iteration


class MACS(Logger):
    """Model-Agnostic Constraint Satisfaction algorithm core.

    Args:
        learner: a `Learner` instance.
        master: a `Master` instance.
        init_step: the initial step of the algorithm, either 'pretraining' or 'projection'.
        metrics: a list of `Metric` instances to evaluate the final solution.
    """

    def __init__(self,
                 learner: Learner,
                 master: Master,
                 init_step: str = 'pretraining',
                 metrics: Opt[List[Metric]] = None):
        super(MACS, self).__init__()
        assert init_step in ['pretraining', 'projection'], f"'{initial_step}' is not a valid initial step"
        self.learner: Learner = learner
        self.master: Master = master
        self.init_step: str = init_step
        self.metrics: List[Metric] = [] if metrics is None else metrics
        self.history: History = History()
        self.fitted: bool = False
        self.time: Opt[float] = None

    def fit(self, x: Matrix, y: Vector, iterations: int = 1, val_data: Opt[Dataset] = None,
            callbacks: Opt[List[Callback]] = None, verbose: Union[int, bool] = 2) -> History:
        """Starts the Moving Target algorithm by iteratively learning and constraining the predictions for the given
           number of iterations using the learner and master instances.

        Args:
            x: the matrix/dataframe of training samples.
            y: the vector of training labels.
            iterations: the number of algorithm iterations.
            val_data: a dictionary containing the validation data, indicated as a tuple (xv, yv).
            callbacks: a list of `Callback` instances.
            verbose: either a boolean or an int representing the verbosity value.

        Returns:
            An instance of the `History` object containing the training history.

        Raises:
            `AssertionError` if the number of iteration is negative, or if zero and the initial step is 'pretraining'.
        """
        # check user input
        assert iterations >= 0, f'the number of iterations should be non-negative, but it is {iterations}'
        assert self.init_step == 'pretraining' or iterations > 0, 'if projection, iterations cannot be zero'
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
        kwargs: Dict = dict(x=x, y=y, val_data=val_data, iteration=0)
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

    def predict(self, x: Matrix) -> Vector:
        """Uses the learner to predict labels from input samples.

        Args:
            x: the matrix/dataframe of input samples.

        Returns:
            The vector of predicted labels.

        Raises:
            `AssertionError` if the learner has not been fitted yet.
        """
        assert self.fitted, 'The model has not been fitted yet, please call method .fit()'
        return self.learner.predict(x)

    def evaluate(self, x: Matrix, y: Vector) -> Dict[str, float]:
        """Evaluates the performances of the model based on the given set of metrics.

        Args:
            x: the matrix/dataframe of training samples.
            y: the vector of training labels.

        Returns:
            The dictionary of evaluated metrics.
        """
        p = self.predict(x)
        return {metric.__name__: metric(x, y, p) for metric in self.metrics}

    # noinspection PyMissingOrEmptyDocstring
    def on_iteration_start(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        self.time = time.time()

    # noinspection PyMissingOrEmptyDocstring
    def on_iteration_end(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
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
