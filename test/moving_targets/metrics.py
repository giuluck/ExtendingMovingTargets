import unittest
from typing import Callable, Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, \
    mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelBinarizer

from moving_targets.metrics import Metric, CrossEntropy, Precision, Recall, F1, Accuracy, AUC, MSE, MAE, R2, \
    ClassFrequenciesStd, MonotonicViolation, DIDI

SEED: int = 0
"""The chosen random seed."""

NUM_SAMPLES: int = 100
"""The number of data points."""

NUM_TESTS: int = 10
"""The number of tests carried out for the same loss."""

NUM_CLASSES: int = 10
"""The number of class labels for multi-class classification tests."""

PLACES: int = 3
"""The number of digits passed to the `assertAlmostEqual()` method."""

METRICS: Dict[str, Union[Tuple[Metric, Callable], Dict[str, Tuple[Metric, Callable]]]] = {
    'crossentropy': (CrossEntropy(), log_loss),
    'precision': {
        'binary': (Precision(), lambda y, p: precision_score(y, p.round().astype(int))),
        'multi': (Precision(average='weighted'), lambda y, p: precision_score(y, p.argmax(axis=1), average='weighted'))
    },
    'recall': {
        'binary': (Recall(), lambda y, p: recall_score(y, p.round().astype(int))),
        'multi': (Recall(average='weighted'), lambda y, p: recall_score(y, p.argmax(axis=1), average='weighted'))
    },
    'f1': {
        'binary': (F1(), lambda y, p: f1_score(y, p.round().astype(int))),
        'multi': (F1(average='weighted'), lambda y, p: f1_score(y, p.argmax(axis=1), average='weighted'))
    },
    'accuracy': {
        'binary': (Accuracy(), lambda y, p: accuracy_score(y, p.round().astype(int))),
        'multi': (Accuracy(), lambda y, p: accuracy_score(y, p.argmax(axis=1))),
    },
    'auc': {
        'binary': (AUC(), roc_auc_score),
        'multi': (AUC(), lambda y, p: roc_auc_score(y, p, multi_class='ovo')),
    },
    'mse': (MSE(), mean_squared_error),
    'mae': (MAE(), mean_absolute_error),
    'r2': (R2(), r2_score)
}
"""Data structure containing the losses to test and the respective scikit losses as ground truths."""


class TestMetrics(unittest.TestCase):
    """Template class to test the correctness of the implemented metrics."""

    def _test(self, metric: str, task: str):
        """The core class to test standard losses.

        :param metric:
            The metric name.

        :param task:
            One in 'regression', 'binary' (for binary classification) and 'multi' (for multiclass classification).
        """
        # fix a random seed for data generation and repeat for the given number of tests
        np.random.seed(SEED)
        for i in range(NUM_TESTS):
            # generates random data (ground truths, predictions, and sample weights).
            if task == 'regression':
                y = np.random.normal(size=NUM_SAMPLES)
                p = np.random.normal(size=NUM_SAMPLES)
            elif task == 'binary':
                y = np.random.randint(0, 2, size=NUM_SAMPLES)
                p = np.random.random(size=NUM_SAMPLES)
            elif task == 'multi':
                y = np.random.randint(0, NUM_CLASSES, size=NUM_SAMPLES)
                p = np.random.random(size=(NUM_SAMPLES, NUM_CLASSES))
                p = p / p.sum(axis=1, keepdims=1)
            else:
                raise ValueError(f"Task '{task}' is not supported")
            # check correctness respectively to scikit metric
            metrics = METRICS[metric]
            mt_metric, sk_metric = metrics if isinstance(metrics, tuple) else metrics[task]
            mt_value = mt_metric(x=[], y=y, p=p)
            sk_value = sk_metric(y, p)
            self.assertAlmostEqual(sk_value, mt_value, places=PLACES, msg=f'iteration: {i}')

    def test_binary_crossentropy(self):
        """Tests the binary crossentropy."""
        self._test(metric='crossentropy', task='binary')

    def test_multi_crossentropy(self):
        """Tests the categorical crossentropy."""
        self._test(metric='crossentropy', task='multi')

    def test_binary_precision(self):
        """Tests the binary precision."""
        self._test(metric='precision', task='binary')

    def test_multi_precision(self):
        """Tests the categorical precision."""
        self._test(metric='precision', task='multi')

    def test_binary_recall(self):
        """Tests the binary recall."""
        self._test(metric='recall', task='binary')

    def test_multi_recall(self):
        """Tests the categorical recall."""
        self._test(metric='recall', task='multi')

    def test_binary_f1(self):
        """Tests the binary f1 score."""
        self._test(metric='f1', task='binary')

    def test_multi_f1(self):
        """Tests the categorical f1 score."""
        self._test(metric='f1', task='multi')

    def test_binary_accuracy(self):
        """Tests the binary accuracy."""
        self._test(metric='accuracy', task='binary')

    def test_multi_accuracy(self):
        """Tests the categorical accuracy."""
        self._test(metric='accuracy', task='multi')

    def test_binary_auc(self):
        """Tests the binary area under curve score."""
        self._test(metric='auc', task='binary')

    def test_multi_auc(self):
        """Tests the categorical area under curve score."""
        self._test(metric='auc', task='multi')

    def test_mae(self):
        """Tests the mean absolute error."""
        self._test(metric='mae', task='regression')

    def test_mse(self):
        """Tests the mean squared error."""
        self._test(metric='mse', task='regression')

    def test_r2(self):
        """Tests the r2 score."""
        self._test(metric='r2', task='regression')

    def test_class_frequencies_std(self):
        """Tests the class frequencies std constraint satisfaction metric."""
        np.random.seed(SEED)
        for i in range(NUM_TESTS):
            # create a dictionary of data counts per class
            counts = {c: 1 + np.random.randint(NUM_SAMPLES) for c in range(NUM_CLASSES)}
            values = np.array([v for v in counts.values()]) / np.sum([v for v in counts.values()])
            # create vector of classes and shuffle it, then obtain fake probabilities due to metric compatibility
            classes = np.concatenate([c * np.ones(n, dtype=int) for c, n in counts.items()])
            np.random.shuffle(classes)
            probabilities = LabelBinarizer().fit_transform(classes.reshape((-1, 1)))
            # compute metric value and actual value (from the class counts)
            metric_value = ClassFrequenciesStd().__call__(x=[], y=[], p=probabilities)
            actual_value = values.std()
            self.assertAlmostEqual(actual_value, metric_value, places=PLACES, msg=f'iteration: {i}')

    def test_monotonic_violation(self):
        """Tests the monotonic violation constraint satisfaction metric."""
        # computes pairwise differences between data points
        def diff(v):
            return np.array([[vi - vj for vj in v] for vi in v])

        # we consider ascending order, thus the expected monotonicities are computed as the sign of the differences
        def mono(v):
            return np.sign(diff(v))

        np.random.seed(SEED)
        for i in range(NUM_TESTS):
            x = np.random.random(NUM_SAMPLES)
            p = np.random.random(NUM_SAMPLES)
            # violations are computed by getting only positive differences between pairwise predictions and then
            # masking the values having expected decreasing monotonicity (so that the increasing dual is not counted)
            violations = diff(p)[mono(x) == -1]
            violations[violations < 1e-3] = 0.0
            # check correctness of average violation
            actual_avg = violations.mean()
            metric_avg = MonotonicViolation(monotonicities_fn=mono, aggregation='average').__call__(x=x, y=[], p=p)
            self.assertAlmostEqual(actual_avg, metric_avg, places=PLACES, msg=f'iteration: {i}')
            # check correctness of percentage violation
            actual_pct = np.sign(violations).mean()
            metric_pct = MonotonicViolation(monotonicities_fn=mono, aggregation='percentage').__call__(x=x, y=[], p=p)
            self.assertAlmostEqual(actual_pct, metric_pct, places=PLACES, msg=f'iteration: {i}')
            # check correctness of feasibility
            actual_fsb = float(np.all(violations == 0))
            metric_fsb = MonotonicViolation(monotonicities_fn=mono, aggregation='feasible').__call__(x=x, y=[], p=p)
            self.assertAlmostEqual(actual_fsb, metric_fsb, places=PLACES, msg=f'iteration: {i}')

    def test_didi(self):
        """Tests the disparate impact discrimination index constraint satisfaction metric."""
        # consider different protected features to test all the possible inputs
        # and different target features to test all the possible tasks
        data = pd.DataFrame.from_dict({
            'bin_protected': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            'multi_protected': [0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
            'onehot_protected_0': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            'onehot_protected_1': [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            'onehot_protected_2': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            'reg_target': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            'bin_target': [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            'multi_target': [1, 2, 0, 0, 0, 0, 1, 1, 1, 2]
        })
        expected_results = {
            'bin_protected': {
                'reg_target': 5.0,
                'bin_target': 0.8,
                'multi_target': 0.8
            },
            'multi_protected': {
                'reg_target': 7.0,
                'bin_target': 1.6333,
                'multi_target': 1.7666
            },
            'onehot_protected': {
                'reg_target': 7.0,
                'bin_target': 1.6333,
                'multi_target': 1.7666
            }
        }
        for protected, target_dict in expected_results.items():
            for target, didi in target_dict.items():
                x = data.drop(columns=[target])
                y = data[target]
                if 'reg' in target:
                    p = y
                    didi_abs = DIDI(classification=False, protected=protected, percentage=False)
                    didi_per = DIDI(classification=False, protected=protected, percentage=True)
                else:
                    p = LabelBinarizer().fit_transform(y)
                    didi_abs = DIDI(classification=True, protected=protected, percentage=False)
                    didi_per = DIDI(classification=True, protected=protected, percentage=True)
                metric_didi_abs, metric_didi_per = didi_abs(x=x, y=y, p=p), didi_per(x=x, y=y, p=p)
                self.assertAlmostEqual(didi, metric_didi_abs, places=PLACES, msg=f"p: '{protected}', t: '{target}'")
                self.assertAlmostEqual(1.0, metric_didi_per, places=PLACES, msg=f"p: '{protected}', t: '{target}'")
