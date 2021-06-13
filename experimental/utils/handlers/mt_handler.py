"""Moving Targets Handler."""
import time
from typing import Any, Union, List, Optional, Dict, Type, Tuple

import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping, History

from experimental.utils.handlers import AbstractHandler, Fold, setup
from moving_targets.callbacks import WandBLogger, Callback
from moving_targets.metrics import Metric, MonotonicViolation
from src.datasets import AbstractManager
from src.models import MT, MTLearner, MTMaster, MTClassificationMaster, MTRegressionMaster
from src.util.augmentation import get_monotonicities_list
from src.util.dictionaries import merge_dictionaries
from src.util.typing import Augmented


# noinspection PyMissingOrEmptyDocstring
class MTHandler(AbstractHandler):
    def __init__(self,
                 manager: AbstractManager,
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
                 mst_backend: str = 'gurobi',
                 mst_loss_fn: Optional[str] = None,
                 mst_alpha: float = 1.0,
                 mst_learner_weights: str = 'all',
                 mst_learner_omega: float = 1.0,
                 mst_master_omega: Optional[float] = None,
                 mst_eps: float = 1e-3,
                 mst_time_limit: Optional[float] = None,
                 mst_custom_args: Dict = None,
                 plt_figsize=(20, 10),
                 plt_num_columns=4,
                 **kwargs):
        super(MTHandler, self).__init__(manager=manager, **kwargs)
        if mst_master_kind in ['cls', 'classification']:
            self.master_class: Type[MTMaster] = MTClassificationMaster
        elif mst_master_kind in ['reg', 'regression']:
            self.master_class: Type[MTMaster] = MTRegressionMaster
        else:
            raise ValueError(f"'{mst_master_kind}' is not a valid master kind")
        self.iterations: int = mt_iterations
        self.init_step: str = mt_init_step
        self.metrics: Optional[List[Metric]] = [] if mt_metrics is None else mt_metrics
        self.verbose: Union[bool, int] = mt_verbose
        self.aug_args: Dict = dict(num_random=aug_num_random, num_ground=aug_num_ground,
                                   num_augmented=aug_num_augmented)
        self.mono_args: Dict = dict(kind=mnt_kind, errors=mnt_errors)
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
        self.master_args: Dict = dict(
            backend=mst_backend,
            alpha=mst_alpha,
            learner_weights=mst_learner_weights,
            learner_omega=mst_learner_omega,
            master_omega=mst_master_omega,
            eps=mst_eps,
            time_limit=mst_time_limit,
            **({} if mst_loss_fn is None else {'loss_fn': mst_loss_fn}),
            **({} if mst_custom_args is None else mst_custom_args)
        )
        self.plot_args: Dict = dict(figsize=plt_figsize, num_columns=plt_num_columns)

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
            MonotonicViolation(monotonicities=mono, aggregation='average', name='avg. violation'),
            MonotonicViolation(monotonicities=mono, aggregation='percentage', name='pct. violation'),
            MonotonicViolation(monotonicities=mono, aggregation='feasible', name='is feasible')
        ]
        model = MT(
            learner=MTLearner(scalers=scalers, **self.learner_args),
            master=self.master_class(monotonicities=mono, augmented_mask=np.isnan(y[label]), **self.master_args),
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
                   num_folds: int = 1,
                   folds_index: Optional[List[int]] = None,
                   extrapolation: bool = False,
                   fold_verbosity: Union[bool, int] = True,
                   model_verbosity: Union[bool, int] = False,
                   callbacks: Optional[List[Callback]] = None,
                   plot_args: Dict = None,
                   summary_args: Dict = None):
        setup(seed=self.seed)
        callbacks = [] if callbacks is None else callbacks
        iterations = self.iterations if iterations is None else iterations
        fold_verbosity = False if num_folds == 1 else fold_verbosity
        if fold_verbosity is not False:
            print(f'{num_folds}-FOLDS CROSS-VALIDATION STARTED')
        for i, fold in enumerate(self.get_folds(num_folds=num_folds, extrapolation=extrapolation)):
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
