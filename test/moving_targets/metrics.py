import unittest
from typing import Callable, Dict, Tuple, Union

import numpy as np
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, \
    mean_squared_error, mean_absolute_error, r2_score

from moving_targets.metrics import Metric, CrossEntropy, Precision, Recall, F1, Accuracy, AUC, MSE, MAE, R2

SEED: int = 0
"""The chosen random seed."""

NUM_SAMPLES: int = 1000
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
    'auc': (AUC(), roc_auc_score),
    'mse': (MSE(), mean_squared_error),
    'mae': (MAE(), mean_absolute_error),
    'r2': (R2(), r2_score)
}


class TestMetrics(unittest.TestCase):
    """Template class to test the correctness of the implemented metrics."""

    def _test(self, metric: str, task: str):
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
            mt_value = mt_metric([], y, p)
            sk_value = sk_metric(y, p)
            self.assertAlmostEqual(sk_value, mt_value, places=PLACES)

    def test_binary_crossentropy(self):
        self._test(metric='crossentropy', task='binary')

    def test_multi_crossentropy(self):
        self._test(metric='crossentropy', task='multi')

    def test_binary_precision(self):
        self._test(metric='precision', task='binary')

    def test_multi_precision(self):
        self._test(metric='precision', task='multi')

    def test_binary_recall(self):
        self._test(metric='recall', task='binary')

    def test_multi_recall(self):
        self._test(metric='recall', task='multi')

    def test_binary_f1(self):
        self._test(metric='f1', task='binary')

    def test_multi_f1(self):
        self._test(metric='f1', task='multi')

    def test_binary_accuracy(self):
        self._test(metric='accuracy', task='binary')

    def test_multi_accuracy(self):
        self._test(metric='accuracy', task='multi')

    def test_binary_auc(self):
        self._test(metric='crossentropy', task='binary')

    def test_multi_auc(self):
        self._test(metric='crossentropy', task='multi')
