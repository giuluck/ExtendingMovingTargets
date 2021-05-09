import unittest
import numpy as np

from docplex.mp.model import Model as CPModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, precision_score
from tensorflow.keras.utils import to_categorical

from moving_targets.masters import CplexMaster

SEED = 0
NUM_KEYS = 50
NUM_TESTS = 100


def custom_loss(name):
    if name == 'sae':
        return lambda yt, yp, sample_weight: NUM_KEYS * mean_absolute_error(yt, yp, sample_weight=sample_weight)
    elif name == 'sse':
        return lambda yt, yp, sample_weight: NUM_KEYS * mean_squared_error(yt, yp, sample_weight=sample_weight)
    elif name == 'cat pre':
        return lambda yt, yp, sample_weight: precision_score(yt, yp, sample_weight=sample_weight, average='micro')
    else:
        raise ValueError(f'{name} is not a valid loss')


class TestCplexLosses(unittest.TestCase):
    def _test(self, cplex_loss, scikit_loss, classes=None, indicator=False, weights=False):
        # fix a random seed for data generation and repeat for the given number of tests
        np.random.seed(SEED)
        for i in range(NUM_TESTS):
            # generate random data (ground truths, predictions, and sample weights) and fix the model variable to be
            # the same as the ground truths in order to obtain the loss as objective function from the minimization
            model = CPModel()
            sample_weight = np.random.uniform(size=NUM_KEYS) if weights else None
            if classes is None:
                model_variables = model.continuous_var_list(keys=NUM_KEYS, lb=-100, ub=100, name='y')
                model_assignments = np.random.uniform(-100, 100, size=NUM_KEYS)
                numeric_variables = np.random.uniform(-100, 100, size=NUM_KEYS)
                model.add_constraints([v == p for v, p in zip(model_variables, model_assignments)])
            elif classes == 2:
                model_variables = model.binary_var_list(keys=NUM_KEYS, name='y')
                model_assignments = np.random.choice([0, 1], size=NUM_KEYS)
                numeric_variables = np.random.uniform(0, 1, size=NUM_KEYS)
                numeric_variables = np.clip(numeric_variables, a_min=0.001, a_max=0.999)
                model.add_constraints([v == p for v, p in zip(model_variables, model_assignments)])
                if indicator:
                    numeric_variables = np.round(numeric_variables).astype(int)
            elif isinstance(classes, int) and classes > 2:
                model_variables = model.binary_var_matrix(keys1=NUM_KEYS, keys2=classes, name='y')
                model_variables = np.array(list(model_variables.values())).reshape(NUM_KEYS, classes)
                model_assignments = np.random.choice(range(classes), size=NUM_KEYS)
                model_assignments = to_categorical(model_assignments, num_classes=classes).astype(int)
                numeric_variables = np.random.uniform(0, 1, size=(NUM_KEYS, classes))
                numeric_variables = (numeric_variables.transpose() / numeric_variables.sum(axis=1)).transpose()
                numeric_variables = np.clip(numeric_variables, a_min=0.001, a_max=0.999)
                model.add_constraints([v == p for v, p in zip(model_variables.flatten(), model_assignments.flatten())])
                if indicator:
                    model_assignments = numeric_variables.argmax(axis=1)
                    numeric_variables = numeric_variables.argmax(axis=1)
            else:
                raise ValueError('num_classes should be either None or an integer greater than one')
            # handle integer predictions in case of precision metric
            model.minimize(cplex_loss(model, numeric_variables, model_variables, sample_weight=sample_weight))
            cplex_objective = model.solve().objective_value
            scikit_objective = scikit_loss(model_assignments, numeric_variables, sample_weight=sample_weight)
            self.assertAlmostEqual(cplex_objective, scikit_objective, delta=3)

    def test_sae(self):
        self._test(CplexMaster.sum_of_absolute_errors, custom_loss('sae'), classes=None, indicator=False, weights=False)

    def test_sae_weights(self):
        self._test(CplexMaster.sum_of_absolute_errors, custom_loss('sae'), classes=None, indicator=False, weights=True)

    def test_sse(self):
        self._test(CplexMaster.sum_of_squared_errors, custom_loss('sse'), classes=None, indicator=False, weights=False)

    def test_sse_weights(self):
        self._test(CplexMaster.sum_of_squared_errors, custom_loss('sse'), classes=None, indicator=False, weights=True)

    def test_mae(self):
        self._test(CplexMaster.mean_absolute_error, mean_absolute_error, classes=None, indicator=False, weights=False)

    def test_mae_weights(self):
        self._test(CplexMaster.mean_absolute_error, mean_absolute_error, classes=None, indicator=False, weights=True)

    def test_mse(self):
        self._test(CplexMaster.mean_squared_error, mean_squared_error, classes=None, indicator=False, weights=False)

    def test_mse_weights(self):
        self._test(CplexMaster.mean_squared_error, mean_squared_error, classes=None, indicator=False, weights=True)

    def test_bce(self):
        self._test(CplexMaster.binary_crossentropy, log_loss, classes=2, indicator=False, weights=False)

    def test_bce_weights(self):
        self._test(CplexMaster.binary_crossentropy, log_loss, classes=2, indicator=False, weights=True)

    def test_bi(self):
        self._test(CplexMaster.binary_indicator, precision_score, classes=2, indicator=True, weights=False)

    def test_bi_weights(self):
        self._test(CplexMaster.binary_indicator, precision_score, classes=2, indicator=True, weights=True)

    def test_cce(self):
        self._test(CplexMaster.categorical_crossentropy, log_loss, classes=5, indicator=False, weights=False)

    def test_cce_weights(self):
        self._test(CplexMaster.categorical_crossentropy, log_loss, classes=5, indicator=False, weights=True)

    def test_ci(self):
        self._test(CplexMaster.categorical_indicator, custom_loss('cat pre'), classes=5, indicator=True, weights=False)

    def test_ci_weights(self):
        self._test(CplexMaster.categorical_indicator, custom_loss('cat pre'), classes=5, indicator=True, weights=True)
