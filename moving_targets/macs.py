"""Core of the Moving Targets algorithm."""

import time
from typing import List, Dict, Callable, Union, Optional

from moving_targets.callbacks.logger import Logger
from moving_targets.callbacks.file_logger import FileLogger
from moving_targets.callbacks.history import History
from moving_targets.callbacks.console_logger import ConsoleLogger
from moving_targets.callbacks.callback import Callback
from moving_targets.learners.learner import Learner
from moving_targets.masters.master import Master
from moving_targets.metrics.metric import Metric
from moving_targets.util.typing import Matrix, Vector, Dataset, Iteration


class MACS(Logger):
    """Model-Agnostic Constraint Satisfaction.

    This class contains the core algorithm of Moving Targets. It leverages the `Learner` and `Master` instances to
    iteratively refine the predictions during the training phase and, eventually, it evaluates the solutions based on
    the given list of `Metric` objects.
    """

    def __init__(self,
                 learner: Learner,
                 master: Master,
                 init_step: str = 'pretraining',
                 metrics: Optional[List[Metric]] = None):
        """
        :param learner:
            A `Learner` instance.

        :param master:
            A `Master` instance.

        :param init_step:
            The initial step of the algorithm, which can be either 'pretraining' or 'projection'.

        :param metrics:
            A list of `Metric` instances to evaluate the final solution.
        """
        super(MACS, self).__init__()
        assert init_step in ['pretraining', 'projection'], f"'{init_step}' is not a valid initial step"

        self.learner: Learner = learner
        """The `Learner` instance."""

        self.master: Master = master
        """The `Master` instance."""

        self.init_step: str = init_step
        """The initial step of the algorithm, which can be either 'pretraining' or 'projection'."""

        self.fitted: bool = False
        """Whether or not the learner has been fitted at least once."""

        self.metrics: List[Metric] = [] if metrics is None else metrics
        """A list of `Metric` instances to evaluate the final solution."""

        self._history: History = History()
        """The internal `History` object which is returned at the end of the training."""

        self._time: Optional[float] = None
        """An auxiliary variable to keep track of the elapsed time between iterations."""

    def fit(self, x: Matrix, y: Vector, iterations: int = 1, val_data: Optional[Dataset] = None,
            callbacks: Optional[List[Callback]] = None, verbose: Union[int, bool] = 2) -> History:
        """Fits the learner based on the Moving Targets iterative procedure.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param iterations:
            The number of algorithm iterations.

        :param val_data:
            A dictionary containing the validation data, indicated as a tuple (xv, yv).

            .. code-block:: python

                val_data = dict(train=(xtr, ytr), validation=(xvl, yvl), test=(xts, yts))

        :param callbacks:
            A list of `Callback` instances.

        :param verbose:
            Either a boolean or an int representing the verbosity value, such that:

            - `0` or `False` create no logger;
            - `1` creates a simple console logger with elapsed time only;
            - `2` or `True` create a more complete console logger with cached data at the end of each iterations.

        :return:
            An instance of the `History` object containing the training history.

        :raise `AssertionError`:
            If the number of iteration is negative, or is zero and the initial step is 'pretraining'.
        """
        # check user input
        assert iterations >= 0, f'the number of iterations should be non-negative, but it is {iterations}'
        assert self.init_step == 'pretraining' or iterations > 0, 'if projection, iterations cannot be zero'
        assert isinstance(verbose, bool) or (isinstance(verbose, int) and verbose in [0, 1, 2]), 'unknown verbosity'
        val_data = {} if val_data is None else (val_data if isinstance(val_data, dict) else {'val': val_data})

        # handle callbacks and verbosity (test for bool before testing for int, as True/False are instances of int)
        callbacks = [] if callbacks is None else callbacks
        print_callback = []
        if isinstance(verbose, bool):
            print_callback = [FileLogger()] if verbose else []
        elif isinstance(verbose, int):
            print_callback = [FileLogger()] if verbose == 2 else ([ConsoleLogger()] if verbose == 1 else [])
        callbacks = print_callback + callbacks
        self._update_callbacks(callbacks, lambda c: c.on_process_start(macs=self, x=x, y=y, val_data=val_data))

        # handle pretraining
        data_args: Dict = dict(x=x, y=y, val_data=val_data, iteration=0)
        if self.init_step == 'pretraining':
            self._update_callbacks(callbacks, lambda c: c.on_pretraining_start(macs=self, **data_args))
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self.learner.fit(macs=self, x=x, y=y, iteration='pretraining')
            self.fitted = True
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self._update_callbacks(callbacks, lambda c: c.on_pretraining_end(macs=self, **data_args))

        # algorithm core
        for iteration in range(1, iterations + 1):
            data_args['iteration'] = iteration
            self._update_callbacks(callbacks, lambda c: c.on_iteration_start(macs=self, **data_args))
            self._update_callbacks(callbacks, lambda c: c.on_adjustment_start(macs=self, **data_args))
            # ---------------------------------------------- MASTER  STEP ----------------------------------------------
            yj, kw = self.master.adjust_targets(macs=self, x=data_args['x'], y=data_args['y'], iteration=iteration), {}
            if yj is None:
                break  # in case of infeasible model, the training loop is stopped
            elif isinstance(yj, tuple):
                yj, kw = yj
                data_args.update(kw)
            # ---------------------------------------------- MASTER  STEP ----------------------------------------------
            self._update_callbacks(callbacks, lambda c: c.on_adjustment_end(macs=self, adjusted_y=yj, **data_args))
            self._update_callbacks(callbacks, lambda c: c.on_training_start(macs=self, **data_args))
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self.learner.fit(macs=self, y=yj, **{k: v for k, v in data_args.items() if k not in ['val_data', 'y']})
            self.fitted = True
            # ---------------------------------------------- LEARNER STEP ----------------------------------------------
            self._update_callbacks(callbacks, lambda c: c.on_training_end(macs=self, **data_args))
            self._update_callbacks(callbacks, lambda c: c.on_iteration_end(macs=self, **data_args))
        self._update_callbacks(callbacks, lambda c: c.on_process_end(macs=self, val_data=val_data))
        return self._history

    def predict(self, x: Matrix) -> Vector:
        """Uses the previously trained `Learner` to predict labels from input samples.

        :param x:
            The matrix/dataframe of input samples.

        :return:
            The vector of predicted labels.

        :raise `AssertionError`:
            If the learner has not been fitted yet.
        """
        assert self.fitted, 'The model has not been fitted yet, please call method .fit()'
        return self.learner.predict(x)

    def evaluate(self, x: Matrix, y: Vector) -> Dict[str, float]:
        """Evaluates the performances of the model based on the given set of metrics.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :return:
            The dictionary of evaluated metrics.

        :raise `AssertionError`:
            If the learner has not been fitted yet.
        """
        p = self.predict(x)
        return {metric.__name__: metric(x, y, p) for metric in self.metrics}

    def on_iteration_start(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration,
                           **additional_kwargs):
        self._time = time.time()

    def on_iteration_end(self, macs, x: Matrix, y: Vector, val_data: Optional[Dataset], iteration: Iteration,
                         **additional_kwargs):
        logs = {'iteration': iteration, 'elapsed time': time.time() - self._time}
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
        """Runs the given routine for each one of the given callbacks, plus the routine for the `MACS` object itself
        (which is run at the beginning) and the inner `History` callback (which is run at the end).

        :param callbacks:
            The list of callbacks specified during the `fit()` call.

        :param routine:
            The callback routine (e.g., <callback>.on_iteration_start()).
        """
        # run callback routine for macs object itself and history object
        routine(self)
        # run callback routine for the history logger and for each external callback
        for callback in callbacks + [self._history]:
            # if the callback is a logger, log the internal cache before calling the routine (and eventually clear it)
            if isinstance(callback, Logger):
                callback.log(**self._cache)
            routine(callback)
        self._cache = {}
