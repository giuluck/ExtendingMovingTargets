"""Experiment Handler."""
import re
import time
from typing import Dict, Any, Union, List, Optional, Callable, Tuple

import wandb

from experimental.utils.configuration import Fold, default_config, setup
from moving_targets import MACS
from moving_targets.callbacks import History, Callback, WandBLogger
from moving_targets.learners import Learner
from moving_targets.masters import Master
from src.datasets import AbstractManager


class ExperimentHandler:
    """Abstract model handler."""

    def __init__(self,
                 manager: AbstractManager,
                 learner: Learner,
                 master: Master,
                 iterations: int = 1,
                 init_step: str = 'pretraining',
                 verbose: Union[int, bool] = False,
                 dataset: Optional[str] = None,
                 wandb_name: Optional[str] = None,
                 wandb_project: Optional[str] = 'moving_targets',
                 wandb_entity: Optional[str] = 'giuluck',
                 wandb_config: Callable = default_config,
                 seed: int = 0):
        """
        :param manager:
            The dataset manager.

        :param learner:
            The `Learner` instance of Moving Targets.

        :param master:
            The `Master` instance of Moving Targets.

        :param iterations:
            The number of Moving Targets iterations.

        :param init_step:
            The Moving Targets initial step, either 'pretraining' or 'projection'.

        :param verbose:
            Either a boolean or an int representing the verbosity value, such that:

            - `0` or `False` create no logger;
            - `1` creates a simple console logger with elapsed time only;
            - `2` or `True` create a more complete console logger with cached data at the end of each iterations.

        :param dataset:
            The dataset name.

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
        """

        def _get_name_from_manager() -> str:
            """If no explicit dataset name is passed, this is retrieved from the dataset manager class name since the
            dataset manager class matches the pattern "<capitalized_dataset_name>Manager", thus we can split by capital
            letters and remove the first (empty) string and the last ("Manager") string, while joining the central part.

            :return:
                The dataset name obtained from the dataset manager class name.
            """
            return ' '.join(re.split('(?=[A-Z])', self.manager.__class__.__name__)[1:-1]).lower()

        self.manager: AbstractManager = manager
        """The dataset manager."""

        self.learner: Learner = learner
        """The `Learner` instance of Moving Targets."""

        self.master: Master = master
        """The `Master` instance of Moving Targets."""

        self.iterations: int = iterations
        """The number of Moving Targets iterations."""

        self.init_step: str = init_step
        """The Moving Targets initial step, either 'pretraining' or 'projection'."""

        self.verbose: Union[bool, int] = verbose
        """Either a boolean or an int representing the verbosity value."""

        self.dataset: str = _get_name_from_manager() if dataset is None else dataset
        """The dataset name."""

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

    def _fit(self, fold: Fold, iterations: int, callbacks: List[Callback],
             verbose: Union[bool, int]) -> Tuple[MACS, History]:
        model = MACS(
            learner=self.learner,
            master=self.master,
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
                if isinstance(c, WandBLogger) and num_folds is not None:
                    c.config['fold'] = i
            # fit model
            model, history = self._fit(fold=fold, iterations=iterations, callbacks=callbacks, verbose=model_verbosity)
            # handle plots
            if plot_args is not None:
                history.plot(**plot_args)
            if summary_args is not None:
                self.manager.evaluation_summary(model, **fold.validation, **summary_args)
            # handle verbosity
            if fold_verbosity in [2, True]:
                print(f'-- elapsed time: {time.time() - start_time}')
        if fold_verbosity is not False:
            print(f'{num_folds}-FOLDS CROSS-VALIDATION ENDED')
