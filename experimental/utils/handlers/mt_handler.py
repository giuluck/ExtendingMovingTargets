"""Moving Targets Handler."""
import time
from typing import Any, Union, List, Optional, Dict, Tuple, Callable

import numpy as np
import pandas as pd
from moving_targets.callbacks import WandBLogger, Callback
from moving_targets.metrics import Metric, MonotonicViolation
from tensorflow.python.keras.callbacks import EarlyStopping, History

from experimental.utils.handlers import AbstractHandler, Fold, setup, default_config
from src.datasets import AbstractManager
from src.models import MT, MTLearner, MTMaster
from src.util.augmentation import get_monotonicities_list
from src.util.dictionaries import merge_dictionaries
from src.util.typing import Augmented


class MTHandler(AbstractHandler):
    """Moving Targets Model Handler."""

    def __init__(self,
                 manager: AbstractManager,
                 model: Optional[str] = None,
                 dataset: Optional[str] = None,
                 wandb_name: Optional[str] = None,
                 wandb_project: Optional[str] = 'moving_targets',
                 wandb_entity: Optional[str] = 'giuluck',
                 wandb_config: Callable = default_config,
                 seed: int = 0,
                 aug_num_augmented: Optional[Augmented] = None,
                 aug_num_random: int = 0,
                 aug_num_ground: Optional[int] = None,
                 mnt_kind: str = 'group',
                 mnt_errors: str = 'raise',
                 mt_iterations: int = 1,
                 mt_init_step: str = 'pretraining',
                 mt_metrics: Optional[List[Metric]] = None,
                 mt_verbose: Union[int, bool] = False,
                 lrn_loss: Optional[str] = None,
                 lrn_optimizer: str = 'adam',
                 lrn_output_act: Optional[str] = None,
                 lrn_h_units: Optional[List[int]] = None,
                 lrn_epochs: int = 200,
                 lrn_batch_size: int = 32,
                 lrn_verbose: bool = False,
                 lrn_early_stop: Optional[EarlyStopping] = EarlyStopping(monitor='loss', patience=10, min_delta=1e-4),
                 lrn_warm_start: bool = False,
                 mst_master_kind: Optional[str] = None,
                 mst_y_loss: str = 'default',
                 mst_p_loss: str = 'default',
                 mst_alpha: float = 1.0,
                 mst_beta: Optional[float] = None,
                 mst_learner_weights: str = 'all',
                 mst_learner_omega: float = 1.0,
                 mst_master_omega: Optional[float] = None,
                 mst_eps: float = 1e-3,
                 plt_figsize=(20, 10),
                 plt_num_columns=4,
                 **mst_kwargs):
        """
        :param manager:
            The dataset manager.

        :param loss:
            The neural network loss function.

        :param model:
            The machine learning model name.

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

        :param aug_num_augmented:
            The number of augmented samples.

        :param aug_num_random:
            The number of unlabelled random samples added to the original dataset.

        :param aug_num_ground:
            The number of samples taken from the original dataset (the remaining ones are ignored).

        :param mnt_kind:
            The monotonicity computation modality:

            - 'ground', which computes the monotonicity within each subgroup respectively to the ground index only.
            - 'group', which computes the monotonicity within each subgroup between each pair in the subgroup.
            - 'all', which computes the monotonicity between each pair in the whole dataset (very slow).

        :param mnt_errors:
            Error strategy when dropping columns due to monotonicity computation.

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

        :param lrn_loss:
            The learner loss function.

        :param lrn_optimizer:
            The learner optimizer.

        :param lrn_output_act:
        The learner output activation.

        :param lrn_h_units:
        The list of learner hidden units.

        :param lrn_epochs:
        The number of learner's training epochs.

        :param lrn_batch_size:
        The batch size for the learner's training.

        :param lrn_verbose:
        Whether or not to print information during the learner's training.

        :param lrn_early_stop:
            An (optional) `EarlyStopping` object for the learner.

        :param lrn_warm_start:
            Whether or not to restart the learner's weight after each moving_targets iteration.

        :param mst_master_kind:
            The Moving Targets' master kind, either 'regression' or 'classification'.

        :param mst_loss_fn:
            The master's loss function.

        :param mst_alpha:
            The non-negative real number which is used to calibrate the two losses in the master's alpha step.

        :param mst_beta:
            The non-negative real number which is used to constraint the p_loss in the master's beta step.

        :param mst_learner_weights:
            The master's learner weights policy, either 'all' or 'infeasible'.

        :param mst_learner_omega:
            Real number that decides the weight of augmented samples during the learning step.

        :param mst_master_omega:
            Real number that decides the weight of augmented samples during the master step.

        :param mst_eps:
            The slack value under which a violation is considered to be acceptable in the master step.

        :param plt_figsize:
            The figsize parameter passed to `plt()`.

        :param plt_num_columns:
            The number of columns of the subplot.

        :param mst_kwargs:
            Any other specific argument to be passed to the super class (i.e., `CplexMaster`, `GurobiMaster`, or
            `CvxpyMaster` depending on the chosen backend).
        """
        super(MTHandler, self).__init__(manager=manager,
                                        model=model,
                                        dataset=dataset,
                                        wandb_name=wandb_name,
                                        wandb_project=wandb_project,
                                        wandb_entity=wandb_entity,
                                        wandb_config=wandb_config,
                                        seed=seed)
        self.classification: bool = True
        """Whether to use a classification or regression `MTMaster`"""

        if mst_master_kind in ['cls', 'classification']:
            self.classification = True
        elif mst_master_kind in ['reg', 'regression']:
            self.classification = False
        else:
            raise ValueError(f"'{mst_master_kind}' is not a valid master kind")

        self.iterations: int = mt_iterations
        """The number of Moving Targets iterations."""

        self.init_step: str = mt_init_step
        """The Moving Targets initial step, either 'pretraining' or 'projection'."""

        self.metrics: Optional[List[Metric]] = [] if mt_metrics is None else mt_metrics
        """A list of `Metric` instances to evaluate the final Moving Targets solution."""

        self.verbose: Union[bool, int] = mt_verbose
        """Either a boolean or an int representing the verbosity value."""

        self.aug_args: Dict = dict(num_random=aug_num_random,
                                   num_ground=aug_num_ground,
                                   num_augmented=aug_num_augmented)
        """Arguments to be passed to the method `AbstractManager.get_augmented_data()`."""

        self.mono_args: Dict = dict(kind=mnt_kind, errors=mnt_errors)
        """Arguments to be passed to the method `get_monotonicities_list()`."""

        self.learner_args: Dict = dict(
            loss=lrn_loss,
            optimizer=lrn_optimizer,
            output_act=lrn_output_act,
            h_units=lrn_h_units,
            epochs=lrn_epochs,
            batch_size=lrn_batch_size,
            verbose=lrn_verbose,
            callbacks=[] if lrn_early_stop is None else [lrn_early_stop],
            warm_start=lrn_warm_start
        )
        """Arguments to be passed to the `MTLearner` constructor."""

        if mst_y_loss != 'default':
            mst_kwargs['y_loss'] = mst_y_loss
        if mst_p_loss != 'default':
            mst_kwargs['p_loss'] = mst_p_loss

        self.master_args: Dict = dict(
            alpha=mst_alpha,
            beta=mst_beta,
            learner_weights=mst_learner_weights,
            learner_omega=mst_learner_omega,
            master_omega=mst_master_omega,
            eps=mst_eps,
            **mst_kwargs
        )
        """Arguments to be passed to the `MTMaster` constructor."""

        self.plot_args: Dict = dict(figsize=plt_figsize, num_columns=plt_num_columns)
        """Arguments passed to the method `History.plot()`."""

    def _fit(self, fold: Fold, iterations: int, metrics: bool, callbacks: List[Callback],
             verbose: Union[bool, int]) -> Tuple[MT, History]:
        label = self.manager.label
        (x, y), scalers = self.manager.get_augmented_data(x=fold.x, y=fold.y, monotonicities=False, **self.aug_args)
        mono = get_monotonicities_list(
            data=pd.concat((x, y), axis=1),
            label=self.manager.label,
            compute_monotonicities=self.manager.compute_monotonicities,
            **self.mono_args
        )
        metrics = [] if not metrics else self.metrics + [
            MonotonicViolation(monotonicities_fn=lambda v: mono, aggregation='average', name='avg. violation'),
            MonotonicViolation(monotonicities_fn=lambda v: mono, aggregation='percentage', name='pct. violation'),
            MonotonicViolation(monotonicities_fn=lambda v: mono, aggregation='feasible', name='is feasible')
        ]
        model = MT(
            learner=MTLearner(scalers=scalers, **self.learner_args),
            master=MTMaster(monotonicities=mono,
                            augmented_mask=np.isnan(y[label]),
                            classification=self.classification,
                            **self.master_args),
            init_step=self.init_step,
            metrics=metrics
        )
        history = model.fit(
            x=x,
            y=y[label],
            iterations=iterations,
            val_data=fold.validation,
            callbacks=callbacks,
            verbose=verbose
        )
        return model, history

    def fit(self, fold: Fold) -> Any:
        model, _ = self._fit(fold=fold, iterations=self.iterations, metrics=False, callbacks=[], verbose=self.verbose)
        return model

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
            model, history = self._fit(fold=fold, iterations=iterations, metrics=True,
                                       callbacks=callbacks, verbose=model_verbosity)
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
