"""Experiment Handler."""

import random
import re
import time
from typing import Dict, Any, Union, List, Optional, Callable, Tuple

import numpy as np
import tensorflow as tf
import wandb

from moving_targets import MACS
from moving_targets.callbacks import History, Callback, WandBLogger
from moving_targets.learners import LogisticRegression
from moving_targets.metrics import Metric
from moving_targets.util.typing import Matrix, Dataset, Vector
from src.datasets import AbstractManager
from src.masters.balanced_counts import BalancedCounts
from src.util.dictionaries import merge_dictionaries
from src.util.preprocessing import Scalers, Scaler


def setup(seed: int = 0):
    """Sets the simulation up.

    :param seed:
        The random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def default_config(handler, **config_kwargs) -> Dict:
    """Returns the default configuration of a given model handler.

    This is useful to have a dictionary of all the handler's parameters for logging information even if they are not
    directly passed to the handler itself, as default values are used.

    :param handler:
        The model handler.

    :param config_kwargs:
        Custom config arguments that are added to default ones.

    :return:
        A dictionary containing the default model handler configuration, plus the custom arguments.
    """
    config = dict()
    for k, v in handler.__dict__.items():
        if k in ['manager', 'wandb_args', 'wandb_config']:
            pass
        elif isinstance(v, dict):
            config.update({f"{k.replace('_args', '')}/{kk}": vv for kk, vv in v.items()})
        else:
            config[k] = v
    config.update(config_kwargs)
    return config


class Fold:
    """Data class containing the information of a fold for k-fold cross-validation."""

    def __init__(self, x: Matrix, y: Vector, scalers: Scalers, validation: Dataset):
        """
        :param x:
            The input data.

        :param y:
            The output data.

        :param scalers:
            The tuple of x/y scalers.

        :param validation:
            A shared validation dataset which is common among all the k folds.
        """
        xsc, ysc = scalers
        xsc = Scaler.get_default(x.shape[-1]) if xsc is None else xsc
        ysc = Scaler.get_default(y.shape[-1]) if ysc is None else ysc

        self.x: Matrix = xsc.fit_transform(x)
        """The input data."""

        self.y: Vector = ysc.fit_transform(y)
        """The output data/info."""

        self.scalers: Scalers = scalers
        """The tuple of x/y scalers."""

        self.validation: Dataset = {k: (xsc.transform(x), ysc.transform(y)) for k, (x, y) in validation.items()}
        """A shared validation dataset which is common among all the k folds."""


class ExperimentHandler:
    """Abstract model handler."""

    def __init__(self,
                 manager: AbstractManager,
                 dataset: Optional[str] = None,
                 mt_iterations: int = 1,
                 mt_init_step: str = 'pretraining',
                 mt_metrics: Optional[List[Metric]] = None,
                 mt_verbose: Union[int, bool] = False,
                 mst_alpha: float = 1.0,
                 mst_beta: Optional[float] = 1.0,
                 plt_figsize=(20, 10),
                 plt_num_columns=4,
                 wandb_name: Optional[str] = None,
                 wandb_project: Optional[str] = 'moving_targets',
                 wandb_entity: Optional[str] = 'giuluck',
                 wandb_config: Callable = default_config,
                 seed: int = 0,
                 **mst_kwargs):
        """
        :param manager:
            The dataset manager.

        :param dataset:
            The dataset name.

        :param mt_iterations:
            The number of Moving Targets iterations.

        :param mt_init_step:
            The Moving Targets initial step, either 'pretraining' or 'projection'.

        :param mt_metrics:
            A list of `Metric` instances to evaluate the final MT solution.

        :param mt_verbose:
            Either a boolean or an int representing the verbosity value, such that:

            - `0` or `False` create no logger;
            - `1` creates a simple console logger with elapsed time only;
            - `2` or `True` create a more complete console logger with cached data at the end of each iterations.

        :param mst_alpha:
            The non-negative real number which is used to calibrate the two losses in the master's alpha step.

        :param mst_beta:
            The non-negative real number which is used to constraint the p_loss in the master's beta step.

        :param plt_figsize:
            The figsize parameter passed to `plt()`.

        :param plt_num_columns:
            The number of columns of the subplot.

        :param wandb_name:
            The Weights&Biases run name. If None, no Weights&Biases instance is created.

        :param wandb_project:
            The Weights&Biases project name. If wandb_name is None, this is ignored.

        :param wandb_entity:
            The Weights&Biases entity name. If wandb_name is None, this is ignored.

        :param wandb_config:
            The Weights&Biases configuration function which is in charge of returning the configuration dictionary.

        :param seed:
            The random seed.
            
        :param mst_kwargs:
            Any other argument to be passed to the `Master` class which is task-specific (e.g., use_prob).
        """

        self.manager = manager
        """The dataset manager."""

        self.dataset: Optional[str] = dataset
        """The dataset name."""

        self.iterations: int = mt_iterations
        """The number of Moving Targets iterations."""

        self.init_step: str = mt_init_step
        """The Moving Targets initial step, either 'pretraining' or 'projection'."""

        self.metrics: Optional[List[Metric]] = [] if mt_metrics is None else mt_metrics
        """A list of `Metric` instances to evaluate the final Moving Targets solution."""

        self.verbose: Union[bool, int] = mt_verbose
        """Either a boolean or an int representing the verbosity value."""

        self.master_args: Dict = dict(alpha=mst_alpha, beta=mst_beta, **mst_kwargs)
        """Arguments to be passed to the `MTMaster` constructor."""

        self.plot_args: Dict = dict(figsize=plt_figsize, num_columns=plt_num_columns)
        """Arguments passed to the method `History.plot()`."""

        self.seed: int = seed
        """The random seed."""

        self.wandb_config: Callable = wandb_config
        """The Weights&Biases configuration function which is in charge of returning the configuration dictionary."""

        self.wandb_args: Optional[Dict] = None if wandb_name is None else dict(
            name=wandb_name,
            project=wandb_project,
            entity=wandb_entity
        )
        """The Weights&Biases instance arguments."""

        # if no explicit dataset name is passed, this is retrieved from the dataset manager class name
        if dataset is None:
            # class name, split by capital letters
            # first empty string (classes begin with capitals) and final 'Test' string are removed
            # the result is joined with spaces and lower-cased
            self.dataset = ' '.join(re.split('(?=[A-Z])', self.manager.__class__.__name__)[1:-1]).lower()

    def _fit(self, fold: Fold, iterations: int, callbacks: List[Callback],
             verbose: Union[bool, int]) -> Tuple[MACS, History]:
        model = MACS(
            learner=LogisticRegression(),
            master=BalancedCounts(**self.master_args),
            init_step=self.init_step,
            metrics=self.manager.metrics
        )
        history = model.fit(
            x=fold.x,
            y=fold.y,
            iterations=iterations,
            val_data=fold.validation,
            callbacks=callbacks,
            verbose=verbose
        )
        return model, history

    def fit(self, fold: Fold) -> Any:
        """Fits the model on the given fold.

        :param fold:
            A fold for k-fold cross-validation.

        :return:
            The machine learning model itself.
        """
        model, _ = self._fit(fold=fold, iterations=self.iterations, callbacks=[], verbose=self.verbose)
        return model

    def get_folds(self, num_folds: Optional[int]) -> Union[Fold, List[Fold]]:
        """Builds the k folds for k-fold cross-validation.

        :param num_folds:
            The number of folds. If None, train/test split only is performed.

        :return:
            A list of folds (or a single fold if num_folds is None).
        """
        folds: List[Fold] = []
        if num_folds is None:
            data, scalers = self.manager.get_folds(num_folds=None)
            x, y = data['train']
            return Fold(x=x, y=y, scalers=scalers, validation=data)
        else:
            for data, scalers in self.manager.get_folds(num_folds=num_folds):
                x, y = data['train']
                fold = Fold(x=x, y=y, scalers=scalers, validation=data)
                folds.append(fold)
        return folds

    def validate(self,
                 num_folds: int = 10,
                 folds_index: Optional[List[int]] = None,
                 summary_args: Optional[Dict] = None):
        """Runs a k-fold cross-validation experiment.

        :param num_folds:
            The number of folds. If None, train/test split only is performed.

        :param folds_index:
            The list of folds index to be validated. If None, all of them are validated.

        :param summary_args:
            The dictionary of arguments for the evaluation summary.
        """
        for i, fold in enumerate(self.get_folds(num_folds=num_folds)):
            if folds_index is None or i in folds_index:
                self._run_instance(fold=fold, index=i, summary_args=summary_args)

    def test(self, summary_args: Optional[Dict] = None):
        """Runs a train/test experiment.

        :param summary_args:
            The dictionary of arguments for the evaluation summary.
        """
        fold = self.get_folds(num_folds=None)
        self._run_instance(fold=fold, index='test', summary_args=summary_args)

    def _run_instance(self, fold: Fold, index: Union[int, str], summary_args: Dict):
        """Runs an experiment instance.

        :param fold:
            A fold for k-fold cross-validation.

        :param index:
            The fold index.

        :param summary_args:
            The dictionary of arguments for the summary plot. If None, no evaluation summary is called.
        """
        setup(seed=self.seed)
        start_time = time.time()
        model = self.fit(fold=fold)
        elapsed_time = time.time() - start_time
        if self.wandb_args is not None:
            wandb.init(**self.wandb_args, config=self.wandb_config(self, fold=index))
            summary = self.manager.evaluation_summary(model, do_print=False, **fold.validation).to_dict()
            wandb.log({
                'elapsed_time': elapsed_time,
                **{f'{split}/{metric}': val for split, metrics in summary.items() for metric, val in metrics.items()}
            })
            wandb.finish()
        if summary_args is not None:
            self.manager.evaluation_summary(model, **fold.validation, **summary_args)

    def experiment(self,
                   iterations: Optional[int] = None,
                   num_folds: Optional[int] = None,
                   folds_index: Optional[List[int]] = None,
                   fold_verbosity: Union[bool, int] = True,
                   model_verbosity: Union[bool, int] = False,
                   callbacks: Optional[List[Callback]] = None,
                   plot_args: Dict = None,
                   summary_args: Dict = None):
        """Builds a more controllable Moving Targets experiment with custom callbacks and plots.

        :param iterations:
            Custom number of Moving Targets' iterations that overwrites the default operation if not None.

        :param num_folds:
            The number of folds. If None, train/test split only is performed.

        :param folds_index:
            The list of folds index to be validated. If None, all of them are validated.

        :param fold_verbosity:
            How much information about the k-fold cross-validation process to print.

        :param model_verbosity:
            How much information about the Moving Targets process to print.

        :param callbacks:
            List of `Callback` object for the Moving Targets process.

        :param plot_args:
            Arguments passed to the method `History.plot()`.

        :param summary_args:
            The dictionary of arguments for the evaluation summary.
        """
        callbacks = [] if callbacks is None else callbacks
        iterations = self.iterations if iterations is None else iterations
        fold_verbosity = False if num_folds is None or num_folds == 1 else fold_verbosity
        if fold_verbosity is not False:
            print(f'{num_folds}-FOLDS CROSS-VALIDATION STARTED')
        folds = self.get_folds(num_folds=num_folds)
        for i, fold in enumerate([folds] if num_folds is None else folds):
            setup(seed=self.seed)
            if folds_index is not None and i not in folds_index:
                continue
            # handle verbosity
            start_time = time.time()
            if fold_verbosity in [2, True]:
                print(f'  > fold {i + 1:0{len(str(num_folds))}}/{num_folds}', end=' ')
            # handle wandb callback
            for c in callbacks:
                if isinstance(c, WandBLogger):
                    c.config['fold'] = i
            # fit model
            model, history = self._fit(fold=fold, iterations=iterations, callbacks=callbacks, verbose=model_verbosity)
            # handle plots
            if plot_args is not None:
                plot_args = merge_dictionaries(self.plot_args, plot_args)
                history.plot(**plot_args)
            if summary_args is not None:
                self.manager.evaluation_summary(model, **fold.validation, **summary_args)
            # handle verbosity
            if fold_verbosity in [2, True]:
                print(f'-- elapsed time: {time.time() - start_time}')
        if fold_verbosity is not False:
            print(f'{num_folds}-FOLDS CROSS-VALIDATION ENDED')
