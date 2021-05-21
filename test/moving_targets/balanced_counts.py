"""Balanced Counts Tests."""

import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from moving_targets import MACS
from moving_targets.learners import LogisticRegression
from moving_targets.masters import BalancedCounts
from moving_targets.metrics import Accuracy, ClassFrequenciesStd

SEED = 0
DATASETS = {
    'iris': dict(
        name='iris',
        separator=',',
        class_column='class',
    ),
    'redwine': dict(
        name='redwine',
        separator=';',
        class_column='quality',
    ),
    'whitewine': dict(
        name='whitewine',
        separator=';',
        class_column='quality',
    )
}
RESULTS = {
    'iris-pretraining-false': dict(
        train_acc=0.9464285714285714,
        train_std=0.011135885079684334,
        val_acc=0.8421052631578947,
        val_std=0.10154242897184408
    ),
    'iris-pretraining-true': dict(
        train_acc=0.9464285714285714,
        train_std=0.011135885079684334,
        val_acc=0.8421052631578947,
        val_std=0.10154242897184408
    ),
    'iris-projection-false': dict(
        train_acc=0.9464285714285714,
        train_std=0.011135885079684334,
        val_acc=0.8421052631578947,
        val_std=0.10154242897184408
    ),
    'iris-projection-true': dict(
        train_acc=0.9464285714285714,
        train_std=0.011135885079684334,
        val_acc=0.8421052631578947,
        val_std=0.10154242897184408
    ),
    'redwine-pretraining-false': dict(
        train_acc=0.3336113427856547,
        train_std=0.038514344281136204,
        val_acc=0.3025,
        val_std=0.032328607902118035
    ),
    'redwine-pretraining-true': dict(
        train_acc=0.3336113427856547,
        train_std=0.038514344281136204,
        val_acc=0.3025,
        val_std=0.032328607902118035
    ),
    'redwine-projection-false': dict(
        train_acc=0.32777314428690574,
        train_std=0.03190968466351949,
        val_acc=0.2675,
        val_std=0.03684615161572358
    ),
    'redwine-projection-true': dict(
        train_acc=0.32777314428690574,
        train_std=0.03190968466351949,
        val_acc=0.2675,
        val_std=0.03684615161572358
    ),
    'whitewine-pretraining-false': dict(
        train_acc=0.2736182956711135,
        train_std=0.026414227999155503,
        val_acc=0.26448979591836735,
        val_std=0.030365967553862658
    ),
    'whitewine-pretraining-true': dict(
        train_acc=0.2736182956711135,
        train_std=0.026414227999155503,
        val_acc=0.26448979591836735,
        val_std=0.030365967553862658
    ),
    'whitewine-projection-false': dict(
        train_acc=0.28178600598965425,
        train_std=0.02516267335925484,
        val_acc=0.26612244897959186,
        val_std=0.02431815458359045
    ),
    'whitewine-projection-true': dict(
        train_acc=0.28178600598965425,
        train_std=0.02516267335925484,
        val_acc=0.26612244897959186,
        val_std=0.02431815458359045
    )
}


class TestBalancedCounts(unittest.TestCase):
    def _test(self, dataset, init_step, use_prob):
        np.random.seed(SEED)
        # load data
        ds = DATASETS[dataset]
        df = pd.read_csv(f"../../res/{ds['name']}.csv", sep=ds['separator']).sample(frac=1)
        x = df.drop(ds['class_column'], axis=1)
        y = df[ds['class_column']].astype('category').cat.codes.values
        num_classes = len(np.unique(y))
        # train/val split and scaling
        x_train, x_val, y_train, y_val = train_test_split(x, y)
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        # define model pieces
        metrics = [Accuracy(name='acc'), ClassFrequenciesStd(classes=num_classes, name='std')]
        learner = LogisticRegression()
        master = BalancedCounts(n_classes=num_classes)
        model = MACS(learner, master, init_step=init_step, metrics=metrics)
        model.fit(x_train, y_train, iterations=3, verbose=False)
        # test results
        exp_res = RESULTS[f'{dataset}-{init_step}-{str(use_prob).lower()}']
        act_res = dict(train=model.evaluate(x_train, y_train), val=model.evaluate(x_val, y_val))
        for split, act in act_res.items():
            for metric, val in act.items():
                self.assertAlmostEqual(exp_res[f'{split}_{metric}'], val)

    def test_iris_pretraining_no_prob(self):
        self._test('iris', 'pretraining', False)

    def test_iris_pretraining_use_prob(self):
        self._test('iris', 'pretraining', True)

    def test_iris_projection_no_prob(self):
        self._test('iris', 'projection', False)

    def test_iris_projection_use_prob(self):
        self._test('iris', 'projection', True)

    def test_redwine_pretraining_no_prob(self):
        self._test('redwine', 'pretraining', False)

    def test_redwine_pretraining_use_prob(self):
        self._test('redwine', 'pretraining', True)

    def test_redwine_projection_no_prob(self):
        self._test('redwine', 'projection', False)

    def test_redwine_projection_use_prob(self):
        self._test('redwine', 'projection', True)

    def test_whitewine_pretraining_no_prob(self):
        self._test('whitewine', 'pretraining', False)

    def test_whitewine_pretraining_use_prob(self):
        self._test('whitewine', 'pretraining', True)

    def test_whitewine_projection_no_prob(self):
        self._test('whitewine', 'projection', False)

    def test_whitewine_projection_use_prob(self):
        self._test('whitewine', 'projection', True)
