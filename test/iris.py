import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.moving_targets import MACS
from src.moving_targets.learners import LogisticRegression
from src.moving_targets.masters import BalancedCounts
from src.moving_targets.metrics import Accuracy, ClassFrequenciesStd

SEED = 0
RESULTS = {
    'pretraining-false': dict(
        train_acc=0.9464285714285714,
        train_std=0.011135885079684334,
        val_acc=0.8421052631578947,
        val_std=0.10154242897184408
    ),
    'pretraining-true': dict(
        train_acc=0.9464285714285714,
        train_std=0.011135885079684334,
        val_acc=0.8421052631578947,
        val_std=0.10154242897184408
    ),
    'projection-false': dict(
        train_acc=0.9464285714285714,
        train_std=0.011135885079684334,
        val_acc=0.8421052631578947,
        val_std=0.10154242897184408
    ),
    'projection-true': dict(
        train_acc=0.9464285714285714,
        train_std=0.011135885079684334,
        val_acc=0.8421052631578947,
        val_std=0.10154242897184408
    )
}


class TestStringMethods(unittest.TestCase):
    def _test(self, initial_step, use_prob):
        np.random.seed(SEED)
        # load iris data
        df = pd.read_csv('../res/iris.csv').sample(frac=1)
        x = df.drop('class', axis=1).values
        y = df['class'].astype('category').cat.codes.values
        # train/val split and scaling
        scaler = MinMaxScaler()
        x_train, x_val, y_train, y_val = train_test_split(x, y)
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        # define model pieces
        metrics = [Accuracy(name='acc'), ClassFrequenciesStd(num_classes=3, name='std')]
        learner = LogisticRegression()
        master = BalancedCounts(n_classes=3)
        model = MACS(learner, master, initial_step=initial_step, use_prob=use_prob, metrics=metrics)
        model.fit(x_train, y_train, iterations=15)
        # test results
        exp_res = RESULTS[f'{initial_step}-{str(use_prob).lower()}']
        act_res = dict(train=model.evaluate(x_train, y_train), val=model.evaluate(x_val, y_val))
        for split, act in act_res.items():
            for metric, val in act.items():
                self.assertAlmostEqual(exp_res[f'{split}_{metric}'], val)

    def test_pretraining_no_prob(self):
        self._test('pretraining', False)

    def test_pretraining_use_prob(self):
        self._test('pretraining', True)

    def test_projection_no_prob(self):
        self._test('projection', False)

    def test_projection_use_prob(self):
        self._test('projection', True)


if __name__ == '__main__':
    unittest.main()
