import random
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping

from moving_targets.callbacks import Callback
from moving_targets.metrics import MonotonicViolation, MSE, R2, CrossEntropy, Accuracy
from src.models import MTLearner, MT, MTRegressionMaster, MTClassificationMaster
from src.util.augmentation import get_monotonicities_list
from src.util.dictionaries import merge_dictionaries


class TestManager:
    DATA_ARGS = dict()
    AUGMENTED_ARGS = dict()
    MONOTONICITIES_ARGS = dict(kind='group')
    LEARNER_ARGS = dict(optimizer='adam', epochs=0, verbose=False,
                        callbacks=[EarlyStopping(monitor='loss', patience=10, min_delta=1e-4)])
    MASTER_ARGS = dict()
    PLOT_ARGS = dict(figsize=(20, 10), num_columns=4)
    SUMMARY_ARGS = dict()

    @staticmethod
    def setup(seed=0, max_rows=10000, max_columns=10000, width=10000, max_colwidth=10000, float_format='{:.4f}'):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        pd.options.display.max_rows = max_rows
        pd.options.display.max_columns = max_columns
        pd.options.display.width = width
        pd.options.display.max_colwidth = max_colwidth
        pd.options.display.float_format = float_format.format

    def __init__(self, dataset, master_type, init_step='pretraining', metrics=None, data_args=None, augmented_args=None,
                 monotonicities_args=None, learner_args=None, master_args=None, seed=0):
        # PARAMETERS
        metrics = [] if metrics is None else metrics
        data_args = merge_dictionaries(TestManager.DATA_ARGS, data_args)
        augmented_args = merge_dictionaries(TestManager.AUGMENTED_ARGS, augmented_args)
        monotonicities_args = merge_dictionaries(TestManager.MONOTONICITIES_ARGS, monotonicities_args)
        learner_args = merge_dictionaries(TestManager.LEARNER_ARGS, learner_args)
        master_args = merge_dictionaries(TestManager.MASTER_ARGS, master_args)
        self.seed = seed
        # DATA
        self.dataset = dataset
        self.data, _ = dataset.load_data(**data_args)
        (self.x, self.y), self.scalers = self.dataset.get_augmented_data(
            x=self.data['train'][0],
            y=self.data['train'][1],
            **augmented_args,
        )
        self.monotonicities = get_monotonicities_list(
            data=pd.concat((self.x, self.y), axis=1),
            compute_monotonicities=self.dataset.compute_monotonicities,
            label=self.dataset.y_column,
            **monotonicities_args
        )
        self.y = self.y[self.dataset.y_column]
        # MOVING TARGETS
        self.moving_targets = MT(
            learner=MTLearner(scalers=self.scalers, **learner_args),
            master=master_type(monotonicities=self.monotonicities, augmented_mask=np.isnan(self.y), **master_args),
            init_step=init_step,
            metrics=metrics + [
                MonotonicViolation(monotonicities=self.monotonicities, aggregation='average', name='avg. violation'),
                MonotonicViolation(monotonicities=self.monotonicities, aggregation='percentage', name='pct. violation'),
                MonotonicViolation(monotonicities=self.monotonicities, aggregation='feasible', name='is feasible')
            ]
        )

    def fit(self, iterations, callbacks=None, verbose=1, plot_args=None, summary_args=None):
        TestManager.setup(seed=self.seed)
        history = self.moving_targets.fit(
            x=self.x,
            y=self.y,
            iterations=iterations,
            val_data=self.data,
            callbacks=callbacks,
            verbose=verbose
        )
        if plot_args is not None:
            plot_args = merge_dictionaries(TestManager.PLOT_ARGS, plot_args)
            history.plot(**plot_args)
        if summary_args is not None:
            summary_args = merge_dictionaries(TestManager.SUMMARY_ARGS, summary_args)
            self.dataset.evaluation_summary(self.moving_targets, **self.data, **summary_args)


class RegressionTest(TestManager):
    def __init__(self, dataset, augmented_args, monotonicities_args, extrapolation=False, warm_start=False, **kwargs):
        super(RegressionTest, self).__init__(
            dataset=dataset,
            master_type=MTRegressionMaster,
            metrics=[MSE(name='loss'), R2(name='metric')],
            data_args=dict(extrapolation=extrapolation),
            augmented_args=augmented_args,
            monotonicities_args=monotonicities_args,
            learner_args=dict(output_act=None, h_units=[16] * 4, optimizer='adam', loss='mse', warm_start=warm_start),
            **kwargs
        )


class ClassificationTest(TestManager):
    def __init__(self, dataset, augmented_args, monotonicities_args, kind='classification', h_units=(128, 128),
                 evaluation_metric=Accuracy(), warm_start=False, **kwargs):
        if kind == 'classification':
            master_type = MTClassificationMaster
            loss_metric = CrossEntropy(name='loss')
            loss_fn = 'binary_crossentropy'
        elif kind == 'regression':
            master_type = MTRegressionMaster
            loss_metric = MSE(name='loss')
            loss_fn = 'mse'
        else:
            raise ValueError(f"kind should be either 'classes' or 'probabilities'")
        super(ClassificationTest, self).__init__(
            dataset=dataset,
            master_type=master_type,
            metrics=[loss_metric, evaluation_metric],
            data_args=dict(),
            augmented_args=augmented_args,
            monotonicities_args=monotonicities_args,
            learner_args=dict(output_act='sigmoid', h_units=h_units, optimizer='adam', loss=loss_fn,
                              warm_start=warm_start),
            **kwargs
        )


class AnalysisCallback(Callback):
    PRETRAINING = 'PT'
    MARKERS = dict(aug='o', label='X')

    def __init__(self, num_columns=5, sorting_attribute=None, file_signature=None, do_plot=True, **kwargs):
        super(AnalysisCallback, self).__init__()
        self.num_columns = num_columns
        self.sorting_attribute = sorting_attribute
        self.file_signature = file_signature
        self.do_plot = do_plot
        self.plot_kwargs = {'figsize': (20, 10), 'tight_layout': True}
        self.plot_kwargs.update(kwargs)
        self.data = None
        self.iterations = []

    def on_process_start(self, macs, x, y, val_data, **kwargs):
        m = pd.Series(['aug' if m else 'label' for m in macs.master.augmented_mask], name='mask')
        self.data = pd.concat((x.reset_index(drop=True), y.reset_index(drop=True), m), axis=1)

    def on_pretraining_start(self, macs, x, y, val_data, **kwargs):
        kwargs['iteration'] = AnalysisCallback.PRETRAINING
        self.on_iteration_start(macs, x, y, val_data, **kwargs)
        self.on_adjustment_start(macs, x, y, val_data, **kwargs)
        self.on_adjustment_end(macs, x, y, np.ones_like(y) * np.nan, val_data, **kwargs)
        self.on_training_start(macs, x, y, val_data, **kwargs)

    def on_pretraining_end(self, macs, x, y, val_data, **kwargs):
        kwargs['iteration'] = AnalysisCallback.PRETRAINING
        self.on_training_end(macs, x, y, val_data, **kwargs)
        self.on_iteration_end(macs, x, y, val_data, **kwargs)

    def on_iteration_start(self, macs, x, y, val_data, iteration, **kwargs):
        self.iterations.append(iteration)
        self.data[f'y {iteration}'] = y

    def on_process_end(self, macs, val_data, **kwargs):
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

    def plot_function(self, iteration):
        pass


class DistanceAnalysis(AnalysisCallback):
    def __init__(self, ground_only=True, num_columns=1, **kwargs):
        super(DistanceAnalysis, self).__init__(num_columns=num_columns, **kwargs)
        self.ground_only = ground_only
        self.y = None

    def on_pretraining_start(self, macs, x, y, val_data, **kwargs):
        self.y = y.name
        super(DistanceAnalysis, self).on_pretraining_start(macs, x, y, val_data, **kwargs)

    def on_training_end(self, macs, x, y, val_data, iteration, **kwargs):
        self.data[f'pred {iteration}'] = macs.predict(x)

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        self.data[f'adj {iteration}'] = adjusted_y

    def on_process_end(self, macs, val_data, **kwargs):
        if self.ground_only:
            self.data = self.data[self.data['mask'] == 'label']
        super(DistanceAnalysis, self).on_process_end(macs, val_data)

    def plot_function(self, iteration):
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


class BoundsAnalysis(AnalysisCallback):
    def __init__(self, num_columns=1, **kwargs):
        super(BoundsAnalysis, self).__init__(num_columns=num_columns, **kwargs)

    def on_process_start(self, macs, x, y, val_data, **kwargs):
        super(BoundsAnalysis, self).on_process_start(macs, x, y, val_data, **kwargs)
        hi, li = macs.master.higher_indices, macs.master.lower_indices
        self.data['lower'] = self.data.index.map(lambda i: li[hi == i])
        self.data['higher'] = self.data.index.map(lambda i: hi[li == i])

    def on_pretraining_end(self, macs, x, y, val_data, **kwargs):
        pass

    def on_training_end(self, macs, x, y, val_data, iteration, **kwargs):
        self._insert_bounds(macs.predict(x), 'pred', iteration)

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        self._insert_bounds(adjusted_y, 'adj', iteration)

    def _insert_bounds(self, v, label, iteration):
        self.data[f'{label} {iteration}'] = v
        self.data[f'{label} lb {iteration}'] = self.data['lower'].map(lambda i: v[i].max() if len(i) > 0 else None)
        self.data[f'{label} ub {iteration}'] = self.data['higher'].map(lambda i: v[i].min() if len(i) > 0 else None)

    def plot_function(self, iteration):
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
