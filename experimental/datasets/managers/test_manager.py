"""Test Managers & Callbacks."""
import random
import re
import time
from typing import List, Any, Type, Dict, Union, Optional as Opt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from pandas import DataFrame
from tensorflow.python.keras.callbacks import EarlyStopping

from moving_targets.callbacks import Callback, WandBLogger
from moving_targets.metrics import MonotonicViolation, MSE, R2, CrossEntropy, Accuracy, Metric
from moving_targets.util.typing import Matrix, Vector, Dataset, Iteration, Monotonicities
from src.datasets import DataManager
from src.models import MTLearner, MT, MTRegressionMaster, MTClassificationMaster, MTMaster
from src.util.augmentation import get_monotonicities_list
from src.util.dictionaries import merge_dictionaries
from src.util.preprocessing import Scalers
from src.util.typing import Augmented

YInfo = Union[Vector, DataFrame]


# noinspection PyMissingOrEmptyDocstring
class Fold:
    def __init__(self, x: Matrix, y: YInfo, scalers: Scalers, monotonicities: Monotonicities, validation: Dataset):
        self.x: Matrix = x
        self.y: YInfo = y
        self.scalers: Scalers = scalers
        self.monotonicities: Monotonicities = monotonicities
        self.validation: Dataset = validation


# noinspection PyMissingOrEmptyDocstring
class TestManager:
    @staticmethod
    def setup(seed: int = 0,
              max_rows: int = 10000,
              max_columns: int = 10000,
              width: int = 10000,
              max_colwidth: int = 10000,
              float_format: str = '{:.4f}'):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        pd.options.display.max_rows = max_rows
        pd.options.display.max_columns = max_columns
        pd.options.display.width = width
        pd.options.display.max_colwidth = max_colwidth
        pd.options.display.float_format = float_format.format

    def __init__(self,
                 dataset: DataManager,
                 master_kind: str,
                 lrn_loss: str,
                 aug_num_augmented: Opt[Augmented] = None,
                 aug_num_random: int = 0,
                 aug_num_ground: Opt[int] = None,
                 mono_kind: str = 'group',
                 mono_errors: str = 'raise',
                 lrn_optimizer: str = 'adam',
                 lrn_output_act: Opt[str] = None,
                 lrn_h_units: Opt[List[int]] = None,
                 lrn_warm_start: bool = False,
                 lrn_epochs: int = 200,
                 lrn_batch_size: int = 32,
                 lrn_verbose: bool = False,
                 lrn_early_stopping: EarlyStopping = EarlyStopping(monitor='loss', patience=10, min_delta=1e-4),
                 mst_backend: str = 'cplex',
                 mst_loss_fn: Opt[str] = None,
                 mst_alpha: float = 1.0,
                 mst_learner_weights: str = 'all',
                 mst_learner_omega: float = 1.0,
                 mst_master_omega: Opt[float] = None,
                 mst_eps: float = 1e-3,
                 mst_time_limit: Opt[float] = None,
                 mst_custom_args: Dict = None,
                 mt_init_step: str = 'pretraining',
                 mt_metrics: List[Metric] = None,
                 plt_figsize=(20, 10),
                 plt_num_columns=4,
                 seed: int = 0):
        self.dataset: DataManager = dataset
        if master_kind in ['cls', 'classification']:
            self.master_class: Type[MTMaster] = MTClassificationMaster
        elif master_kind in ['reg', 'regression']:
            self.master_class: Type[MTMaster] = MTRegressionMaster
        else:
            raise ValueError(f"'{master_kind}' is not a valid master kind")
        self.augmented_args: Dict = dict(
            num_random=aug_num_random,
            num_ground=aug_num_ground,
            num_augmented=aug_num_augmented
        )
        self.monotonicities_args: Dict = dict(
            kind=mono_kind,
            errors=mono_errors
        )
        self.learner_args: Dict = dict(
            loss=lrn_loss,
            optimizer=lrn_optimizer,
            output_act=lrn_output_act,
            h_units=lrn_h_units,
            warm_start=lrn_warm_start,
            epochs=lrn_epochs,
            batch_size=lrn_batch_size,
            verbose=lrn_verbose,
            callbacks=[lrn_early_stopping]
        )
        self.master_args = dict(
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
        self.mt_init_step: str = mt_init_step
        self.mt_metrics: List[Metric] = [] if mt_metrics is None else mt_metrics
        self.seed: int = seed
        self.plot_args: Dict = dict(
            figsize=plt_figsize,
            num_columns=plt_num_columns
        )
        self.summary_args: Dict = dict()
        # class name, split by capital letters
        # first empty string (classes begin with capitals) and final 'Test' string are removed
        # the result is joined with spaces and lower-cased
        self.name: str = ' '.join(re.split('(?=[A-Z])', self.__class__.__name__)[1:-1]).lower()

    def get_folds(self, num_folds: int, extrapolation: bool, compute_monotonicities: bool = True) -> List[Fold]:
        folds: List[Fold] = []
        for data, _ in self.dataset.load_data(num_folds=num_folds, extrapolation=extrapolation):
            (x_aug, y_aug), scalers = self.dataset.get_augmented_data(
                x=data['train'][0],
                y=data['train'][1],
                monotonicities=compute_monotonicities,
                **self.augmented_args
            )
            monotonicities = get_monotonicities_list(
                data=pd.concat((x_aug, y_aug), axis=1),
                label=self.dataset.y_column,
                compute_monotonicities=self.dataset.compute_monotonicities,
                **self.monotonicities_args
            )
            folds.append(Fold(
                x=x_aug,
                y=y_aug,
                scalers=scalers,
                monotonicities=monotonicities,
                validation=data
            ))
        return folds

    def get_model(self, fold: Fold):
        return MT(
            learner=MTLearner(scalers=fold.scalers, **self.learner_args),
            master=self.master_class(monotonicities=fold.monotonicities,
                                     augmented_mask=np.isnan(fold.y[self.dataset.y_column]),
                                     **self.master_args),
            init_step=self.mt_init_step,
            metrics=self.mt_metrics + [
                MonotonicViolation(monotonicities=fold.monotonicities, aggregation='average', name='avg. violation'),
                MonotonicViolation(monotonicities=fold.monotonicities, aggregation='percentage', name='pct. violation'),
                MonotonicViolation(monotonicities=fold.monotonicities, aggregation='feasible', name='is feasible')
            ]
        )

    def validate(self,
                 iterations: int,
                 num_folds: int = 10,
                 callbacks: Opt[List[Callback]] = None,
                 verbose: Union[bool, int] = True,
                 plot_args: Dict = None,
                 summary_args: Dict = None):
        callbacks = [] if callbacks is None else callbacks
        if verbose is not False:
            print(f'{num_folds}-FOLDS CROSS-VALIDATION STARTED')
        for i, fold in enumerate(self.get_folds(num_folds=num_folds, extrapolation=False)):
            # handle verbosity
            start_time = time.time()
            if verbose in [2, True]:
                print(f'  > fold {i + 1:0{len(str(num_folds))}}/{num_folds}', end=' ')
            # handle wandb callback
            for c in callbacks:
                if isinstance(c, WandBLogger):
                    c.config['fold'] = i
            # fit model
            self._run_instance(fold=fold,
                               iterations=iterations,
                               callbacks=callbacks,
                               verbose=False,
                               plot_args=plot_args,
                               summary_args=summary_args)
            # handle verbosity
            if verbose in [2, True]:
                print(f'-- elapsed time: {time.time() - start_time}')
        if verbose is not False:
            print(f'{num_folds}-FOLDS CROSS-VALIDATION ENDED')

    def test(self,
             iterations: int,
             callbacks: Opt[List[Callback]] = None,
             verbose: Union[int, bool] = 1,
             plot_args: Dict = None,
             summary_args: Dict = None,
             extrapolation: bool = False):
        # get single fold with train/val/test splits
        fold = self.get_folds(num_folds=1, extrapolation=extrapolation)[0]
        # fit model
        self._run_instance(fold=fold,
                           iterations=iterations,
                           callbacks=callbacks,
                           verbose=verbose,
                           plot_args=plot_args,
                           summary_args=summary_args)

    def _run_instance(self,
                      fold: Fold,
                      iterations: int,
                      callbacks: Opt[List[Callback]],
                      verbose: Union[int, bool],
                      plot_args: Dict = None,
                      summary_args: Dict = None):
        TestManager.setup(seed=self.seed)
        model = self.get_model(fold)
        history = model.fit(
            x=fold.x,
            y=fold.y[self.dataset.y_column],
            iterations=iterations,
            val_data=fold.validation,
            callbacks=callbacks,
            verbose=verbose
        )
        if plot_args is not None:
            plot_args = merge_dictionaries(self.plot_args, plot_args)
            history.plot(**plot_args)
        if summary_args is not None:
            summary_args = merge_dictionaries(self.summary_args, summary_args)
            self.dataset.evaluation_summary(model, **fold.validation, **summary_args)


# noinspection PyMissingOrEmptyDocstring
class RegressionTest(TestManager):
    def __init__(self,
                 dataset: DataManager,
                 **kwargs):
        super(RegressionTest, self).__init__(
            dataset=dataset,
            master_kind='regression',
            mt_metrics=[MSE(name='loss'), R2(name='metric')],
            lrn_loss='mse',
            lrn_output_act=None,
            **kwargs
        )


# noinspection PyMissingOrEmptyDocstring
class ClassificationTest(TestManager):
    def __init__(self,
                 dataset: DataManager,
                 master_kind: str = 'classification',
                 mst_evaluation_metric: Metric = Accuracy(),
                 **kwargs):
        if master_kind == 'classification':
            mt_loss_metric = CrossEntropy(name='loss')
            lrn_loss = 'binary_crossentropy'
        elif master_kind == 'regression':
            mt_loss_metric = MSE(name='loss')
            lrn_loss = 'mse'
        else:
            raise ValueError(f"'{master_kind}' is not a valid master kind")
        super(ClassificationTest, self).__init__(
            dataset=dataset,
            master_kind=master_kind,
            mt_metrics=[mt_loss_metric, mst_evaluation_metric],
            lrn_loss=lrn_loss,
            lrn_output_act='sigmoid',
            **kwargs
        )


# noinspection PyMissingOrEmptyDocstring
class AnalysisCallback(Callback):
    PRETRAINING = 'PT'
    MARKERS = dict(aug='o', label='X')

    def __init__(self, num_columns: int = 5, sorting_attribute: object = None, file_signature: str = None,
                 do_plot: bool = True, **kwargs):
        super(AnalysisCallback, self).__init__()
        self.num_columns: int = num_columns
        self.sorting_attribute: object = sorting_attribute
        self.file_signature: str = file_signature
        self.do_plot: bool = do_plot
        self.plot_kwargs: Dict = {'figsize': (20, 10), 'tight_layout': True}
        self.plot_kwargs.update(kwargs)
        self.data: Opt[pd.DataFrame] = None
        self.iterations: List[Any] = []

    def on_process_start(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], **kwargs):
        m = pd.Series(['aug' if m else 'label' for m in macs.master.augmented_mask], name='mask')
        self.data = pd.concat((x.reset_index(drop=True), y.reset_index(drop=True), m), axis=1)

    def on_pretraining_start(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], **kwargs):
        kwargs['iteration'] = AnalysisCallback.PRETRAINING
        self.on_iteration_start(macs, x, y, val_data, **kwargs)
        self.on_adjustment_start(macs, x, y, val_data, **kwargs)
        self.on_adjustment_end(macs, x, y, np.ones_like(y) * np.nan, val_data, **kwargs)
        self.on_training_start(macs, x, y, val_data, **kwargs)

    def on_pretraining_end(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], **kwargs):
        kwargs['iteration'] = AnalysisCallback.PRETRAINING
        self.on_training_end(macs, x, y, val_data, **kwargs)
        self.on_iteration_end(macs, x, y, val_data, **kwargs)

    def on_iteration_start(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration,
                           **kwargs):
        self.iterations.append(iteration)
        self.data[f'y {iteration}'] = y

    def on_process_end(self, macs, val_data: Opt[Dataset], **kwargs):
        # sort values
        if self.sorting_attribute is not None:
            self.data = self.data.sort_values(self.sorting_attribute)
        # write on files
        if self.file_signature is not None:
            self.data.to_csv(self.file_signature + '.csv', index_label='index')
            with open(self.file_signature + '.txt', 'w') as f:
                f.write(str(self.data))
        # do plots
        if self.do_plot:
            plt.figure(**self.plot_kwargs)
            num_rows = int(np.ceil(len(self.iterations) / self.num_columns))
            ax = None
            for idx, it in enumerate(self.iterations):
                ax = plt.subplot(num_rows, self.num_columns, idx + 1, sharex=ax, sharey=ax)
                title = self.plot_function(it)
                ax.set(xlabel='', ylabel='')
                ax.set_title(f'{it})' if title is None else title)
            plt.show()

    def plot_function(self, iteration: Iteration) -> Opt[str]:
        pass


# noinspection PyMissingOrEmptyDocstring
class DistanceAnalysis(AnalysisCallback):
    def __init__(self, ground_only: bool = True, num_columns=1, **kwargs):
        super(DistanceAnalysis, self).__init__(num_columns=num_columns, **kwargs)
        self.ground_only: bool = ground_only
        self.y: Opt[str] = None

    def on_pretraining_start(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], **kwargs):
        self.y = y.name
        super(DistanceAnalysis, self).on_pretraining_start(macs, x, y, val_data, **kwargs)

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        self.data[f'pred {iteration}'] = macs.predict(x)

    def on_adjustment_end(self, macs, x: Matrix, y: Vector, adjusted_y: Vector, val_data: Opt[Dataset],
                          iteration: Iteration, **kwargs):
        self.data[f'adj {iteration}'] = adjusted_y

    def on_process_end(self, macs, val_data: Opt[Dataset], **kwargs):
        if self.ground_only:
            self.data = self.data[self.data['mask'] == 'label']
        super(DistanceAnalysis, self).on_process_end(macs, val_data)

    def plot_function(self, iteration: Iteration) -> Opt[str]:
        x = np.arange(len(self.data))
        y, p, j = self.data[self.y].values, self.data[f'pred {iteration}'].values, self.data[f'adj {iteration}'].values
        style = self.data['mask']
        sns.scatterplot(x=x, y=y, color='black', alpha=0.6).set_xticks([])
        sns.scatterplot(x=x, y=p, color='red', alpha=0.6)
        sns.scatterplot(x=x, y=j, style=style, markers=AnalysisCallback.MARKERS, color='blue', alpha=0.8, s=50)
        plt.legend(['labels', 'predictions', 'adjusted'])
        for i in x:
            plt.plot([i, i], [p[i], j[i]], c='red')
            plt.plot([i, i], [y[i], j[i]], c='black')
        avg_pred_distance = np.abs(p - j).mean()
        avg_label_distance = np.abs(y[style == 'label'] - j[style == 'label']).mean()
        return f'{iteration}) pred. distance = {avg_pred_distance:.4f}, label distance = {avg_label_distance:.4f}'


# noinspection PyMissingOrEmptyDocstring
class BoundsAnalysis(AnalysisCallback):
    def __init__(self, num_columns: int = 1, **kwargs):
        super(BoundsAnalysis, self).__init__(num_columns=num_columns, **kwargs)

    def on_process_start(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], **kwargs):
        super(BoundsAnalysis, self).on_process_start(macs, x, y, val_data, **kwargs)
        hi, li = macs.master.higher_indices, macs.master.lower_indices
        self.data['lower'] = self.data.index.map(lambda i: li[hi == i])
        self.data['higher'] = self.data.index.map(lambda i: hi[li == i])

    def on_pretraining_end(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], **kwargs):
        pass

    def on_training_end(self, macs, x: Matrix, y: Vector, val_data: Opt[Dataset], iteration: Iteration, **kwargs):
        self._insert_bounds(macs.predict(x), 'pred', iteration)

    def on_adjustment_end(self, macs, x: Matrix, y: Vector, adjusted_y: Vector, val_data: Opt[Dataset],
                          iteration: Iteration, **kwargs):
        self._insert_bounds(adjusted_y, 'adj', iteration)

    def _insert_bounds(self, v: np.ndarray, label: str, iteration: Iteration):
        self.data[f'{label} {iteration}'] = v
        self.data[f'{label} lb {iteration}'] = self.data['lower'].map(lambda i: v[i].max() if len(i) > 0 else None)
        self.data[f'{label} ub {iteration}'] = self.data['higher'].map(lambda i: v[i].min() if len(i) > 0 else None)

    def plot_function(self, iteration: Iteration) -> Opt[str]:
        x = np.arange(len(self.data))
        avg_bound = {}
        for label, color in dict(adj='blue', pred='red').items():
            val = self.data[f'{label} {iteration}']
            lb = self.data[f'{label} lb {iteration}'].fillna(val.min())
            ub = self.data[f'{label} ub {iteration}'].fillna(val.max())
            sns.scatterplot(x=x, y=lb, marker='^', color=color, alpha=0.4)
            sns.scatterplot(x=x, y=ub, marker='v', color=color, alpha=0.4)
            sns.scatterplot(x=x, y=val, color=color, edgecolors='black', label=label).set_xticks([])
            avg_bound[label] = np.mean(ub - lb)
        return f'{iteration}) ' + ', '.join([f'{k} bound = {v:.2f}' for k, v in avg_bound.items()])
