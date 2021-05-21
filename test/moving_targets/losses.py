"""Losses Tests."""

import unittest

import numpy as np
from docplex.mp.model import Model as CplexModel
from gurobipy import Model as GurobiModel, Env, GRB
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, precision_score
from tensorflow.keras.utils import to_categorical

from moving_targets.masters import CplexMaster, GurobiMaster

SEED = 0
NUM_KEYS = 50
NUM_TESTS = 100


class TestLosses:
    @staticmethod
    def _custom_loss(name):
        if name == 'sae':
            return lambda yt, yp, sample_weight: NUM_KEYS * mean_absolute_error(yt, yp, sample_weight=sample_weight)
        elif name == 'sse':
            return lambda yt, yp, sample_weight: NUM_KEYS * mean_squared_error(yt, yp, sample_weight=sample_weight)
        elif name == 'swapped bce':
            return lambda yt, yp, sample_weight: log_loss(yp, yt, sample_weight=sample_weight)
        elif name == 'categorical precision':
            return lambda yt, yp, sample_weight: precision_score(yt, yp, sample_weight=sample_weight, average='micro')
        else:
            raise ValueError(f'{name} is not a valid loss')

    def _losses(self):
        raise NotImplementedError(f"Please implement method '_losses'")

    def _test(self, model_loss, scikit_loss, classes=None, weights=False):
        raise NotImplementedError(f"Please implement method '_test'")

    def test_sae(self):
        self._test(
            model_loss=self._losses().sum_of_absolute_errors,
            scikit_loss=TestLosses._custom_loss('sae'),
            classes=None,
            weights=False
        )

    def test_sae_weights(self):
        self._test(
            model_loss=self._losses().sum_of_absolute_errors,
            scikit_loss=TestLosses._custom_loss('sae'),
            classes=None,
            weights=True
        )

    def test_sse(self):
        self._test(
            model_loss=self._losses().sum_of_squared_errors,
            scikit_loss=TestLosses._custom_loss('sse'),
            classes=None,
            weights=False
        )

    def test_sse_weights(self):
        self._test(
            model_loss=self._losses().sum_of_squared_errors,
            scikit_loss=TestLosses._custom_loss('sse'),
            classes=None,
            weights=True
        )

    def test_mae(self):
        self._test(
            model_loss=self._losses().mean_absolute_error,
            scikit_loss=mean_absolute_error,
            classes=None,
            weights=False
        )

    def test_mae_weights(self):
        self._test(
            model_loss=self._losses().mean_absolute_error,
            scikit_loss=mean_absolute_error,
            classes=None,
            weights=True
        )

    def test_mse(self):
        self._test(
            model_loss=self._losses().mean_squared_error,
            scikit_loss=mean_squared_error,
            classes=None,
            weights=False
        )

    def test_mse_weights(self):
        self._test(
            model_loss=self._losses().mean_squared_error,
            scikit_loss=mean_squared_error,
            classes=None,
            weights=True
        )

    def test_bh(self):
        self._test(
            model_loss=self._losses().binary_hamming,
            scikit_loss=precision_score,
            classes=(2, 'indicator'),
            weights=False
        )

    def test_bh_weights(self):
        self._test(
            model_loss=self._losses().binary_hamming,
            scikit_loss=precision_score,
            classes=(2, 'indicator'),
            weights=True)

    def test_bce(self):
        self._test(
            model_loss=self._losses().binary_crossentropy,
            scikit_loss=log_loss,
            classes=(2, 'probability'),
            weights=False
        )

    def test_bce_weights(self):
        self._test(
            model_loss=self._losses().binary_crossentropy,
            scikit_loss=log_loss,
            classes=(2, 'probability'),
            weights=True
        )

    def test_swapped_bce(self):
        self._test(
            model_loss=self._losses().swapped_binary_crossentropy,
            scikit_loss=TestLosses._custom_loss('swapped bce'),
            classes=(2, 'swapped'),
            weights=False
        )

    def test_swapped_bce_weights(self):
        self._test(
            model_loss=self._losses().swapped_binary_crossentropy,
            scikit_loss=TestLosses._custom_loss('swapped bce'),
            classes=(2, 'swapped'),
            weights=True
        )

    def test_ch(self):
        self._test(
            model_loss=self._losses().categorical_hamming,
            scikit_loss=TestLosses._custom_loss('categorical precision'),
            classes=(5, 'indicator'),
            weights=False
        )

    def test_ch_weights(self):
        self._test(
            model_loss=self._losses().categorical_hamming,
            scikit_loss=TestLosses._custom_loss('categorical precision'),
            classes=(5, 'indicator'),
            weights=True
        )

    def test_cce(self):
        self._test(
            model_loss=self._losses().categorical_crossentropy,
            scikit_loss=log_loss,
            classes=(5, 'probability'),
            weights=False
        )

    def test_cce_weights(self):
        self._test(
            model_loss=self._losses().categorical_crossentropy,
            scikit_loss=log_loss,
            classes=(5, 'probability'),
            weights=True
        )


class TestCplexLosses(TestLosses, unittest.TestCase):
    def _losses(self):
        return CplexMaster.losses

    def _test(self, model_loss, scikit_loss, classes=None, weights=False):
        # fix a random seed for data generation and repeat for the given number of tests
        np.random.seed(SEED)
        for i in range(NUM_TESTS):
            # generate random data (ground truths, predictions, and sample weights) and fix the model variable to be
            # the same as the ground truths in order to obtain the loss as objective function from the minimization
            model = CplexModel()
            sample_weight = np.random.uniform(size=NUM_KEYS) if weights else None
            if classes is None:
                model_variables = model.continuous_var_list(keys=NUM_KEYS, lb=-100, ub=100, name='y')
                model_assignments = np.random.uniform(-100, 100, size=NUM_KEYS)
                numeric_variables = np.random.uniform(-100, 100, size=NUM_KEYS)
                constraints = [v == p for v, p in zip(model_variables, model_assignments)]
            else:
                num_classes, kind = classes
                if num_classes == 2:
                    model_variables = model.binary_var_list(keys=NUM_KEYS, name='y')
                    model_assignments = np.random.choice([0, 1], size=NUM_KEYS)
                    numeric_variables = np.random.uniform(0.001, 0.999, size=NUM_KEYS)
                    constraints = [v == p for v, p in zip(model_variables, model_assignments)]
                    if kind == 'indicator':
                        numeric_variables = np.round(numeric_variables).astype(int)
                    elif kind != 'probability':
                        raise ValueError(f"unsupported kind '{kind}'")
                elif isinstance(num_classes, int) and num_classes > 2:
                    model_variables = model.binary_var_matrix(keys1=NUM_KEYS, keys2=num_classes, name='y')
                    model_variables = np.array(list(model_variables.values())).reshape(NUM_KEYS, num_classes)
                    model_assignments = np.random.choice(range(num_classes), size=NUM_KEYS)
                    model_assignments = to_categorical(model_assignments, num_classes=num_classes).astype(int)
                    numeric_variables = np.random.uniform(0.001, 0.999, size=(NUM_KEYS, num_classes))
                    numeric_variables = (numeric_variables.transpose() / numeric_variables.sum(axis=1)).transpose()
                    constraints = [v == p for v, p in zip(model_variables.flatten(), model_assignments.flatten())]
                    if kind == 'indicator':
                        model_assignments = numeric_variables.argmax(axis=1)
                        numeric_variables = numeric_variables.argmax(axis=1)
                    elif kind != 'probability':
                        raise ValueError(f"unsupported kind '{kind}'")
                else:
                    raise ValueError('num_classes should be either None or a tuple')
            # handle integer predictions in case of precision metric
            model.add_constraints(constraints)
            model.minimize(model_loss(
                model=model,
                numeric_variables=numeric_variables,
                model_variables=model_variables,
                sample_weight=sample_weight
            ))
            cplex_objective = model.solve().objective_value
            scikit_objective = scikit_loss(model_assignments, numeric_variables, sample_weight=sample_weight)
            self.assertAlmostEqual(cplex_objective, scikit_objective, delta=3)

    def test_swapped_bce(self):
        pass

    def test_swapped_bce_weights(self):
        pass


class TestGurobiLosses(TestLosses, unittest.TestCase):
    def _losses(self):
        return GurobiMaster.losses

    def _test(self, model_loss, scikit_loss, classes=None, weights=False):
        # fix a random seed for data generation and repeat for the given number of tests
        np.random.seed(SEED)
        for i in range(NUM_TESTS):
            # generate random data (ground truths, predictions, and sample weights) and fix the model variable to be
            # the same as the ground truths in order to obtain the loss as objective function from the minimization
            with Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
                with GurobiModel(env=env) as model:
                    sample_weight = np.random.uniform(size=NUM_KEYS) if weights else None
                    if classes is None:
                        model_variables = model.addVars(NUM_KEYS, vtype=GRB.CONTINUOUS, lb=-100, ub=100, name='y')
                        model_variables = np.array(model_variables.values())
                        model_assignments = np.random.uniform(-100, 100, size=NUM_KEYS)
                        numeric_variables = np.random.uniform(-100, 100, size=NUM_KEYS)
                        constraints = (v == p for v, p in zip(model_variables, model_assignments))
                    else:
                        num_classes, kind = classes
                        if num_classes == 2 and kind != 'swapped':
                            model_variables = model.addVars(NUM_KEYS, vtype=GRB.BINARY, name='y')
                            model_variables = np.array(model_variables.values())
                            model_assignments = np.random.choice([0, 1], size=NUM_KEYS)
                            numeric_variables = np.random.uniform(0.001, 0.999, size=NUM_KEYS)
                            constraints = (v == p for v, p in zip(model_variables, model_assignments))
                            if kind == 'indicator':
                                numeric_variables = np.round(numeric_variables).astype(int)
                            elif kind != 'probability':
                                raise ValueError(f"unsupported kind '{kind}'")
                        elif num_classes == 2 and kind == 'swapped':
                            model_variables = model.addVars(NUM_KEYS, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='y')
                            model_variables = np.array(model_variables.values())
                            model_assignments = np.random.uniform(0.001, 0.999, size=NUM_KEYS)
                            numeric_variables = np.random.choice([0, 1], size=NUM_KEYS)
                            constraints = (v == p for v, p in zip(model_variables, model_assignments))
                        elif isinstance(num_classes, int) and num_classes > 2:
                            model_variables = model.addVars(NUM_KEYS, num_classes, vtype=GRB.BINARY, name='y')
                            model_variables = np.array(model_variables.values())
                            model_assignments = np.random.choice(range(num_classes), size=NUM_KEYS)
                            model_assignments = to_categorical(model_assignments, num_classes=num_classes).astype(int)
                            numeric_variables = np.random.uniform(0.001, 0.999, size=(NUM_KEYS, num_classes))
                            numeric_variables = numeric_variables.transpose() / numeric_variables.sum(axis=1)
                            numeric_variables = numeric_variables.transpose()
                            constraints = (v == p for v, p in zip(model_variables, model_assignments.flatten()))
                            model_variables = model_variables.reshape(NUM_KEYS, num_classes)
                            if kind == 'indicator':
                                model_assignments = numeric_variables.argmax(axis=1)
                                numeric_variables = numeric_variables.argmax(axis=1)
                            elif kind != 'probability':
                                raise ValueError(f"unsupported kind '{kind}'")
                        else:
                            raise ValueError('num_classes should be either None or a tuple')
                    # handle integer predictions in case of precision metric
                    model.update()
                    model.addConstrs(constraints, name='c')
                    model.setObjective(model_loss(
                        model=model,
                        numeric_variables=numeric_variables,
                        model_variables=model_variables,
                        sample_weight=sample_weight
                    ), GRB.MINIMIZE)
                    model.optimize()
                    gurobi_objective = model.objVal
                    scikit_objective = scikit_loss(model_assignments, numeric_variables, sample_weight=sample_weight)
                    self.assertAlmostEqual(gurobi_objective, scikit_objective, delta=3)
