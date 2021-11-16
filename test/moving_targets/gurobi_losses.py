import unittest
from typing import Callable, Optional, Tuple

import numpy as np
from gurobipy import Model as GurobiModel, Env, GRB
from tensorflow.keras.utils import to_categorical as one_hot

from moving_targets.masters import GurobiMaster
from test.moving_targets.abstract_losses import TestLosses, SEED, NUM_TESTS, NUM_KEYS, PLACES


class TestGurobiLosses(TestLosses, unittest.TestCase):
    """Tests the correctness of Gurobi losses."""

    def _losses(self):
        """The solver's losses.

        :returns:
            Gurobi's `LossesHandler` object.
        """
        return GurobiMaster.losses

    def _test(self,
              model_loss: Callable,
              scikit_loss: Callable,
              classes: Optional[Tuple[int, str]] = None,
              weights: bool = False):
        """The core class, which must be implemented by the solver so to compare the obtained loss and the ground loss.

        :param model_loss:
            The solver loss.

        :param scikit_loss:
            The ground truth loss (obtained from scikit learn and custom losses).

        :param classes:
            Either None in case of regression losses or a tuple containing the number of classes and the kind of
            postprocessing needed for the predictions which can be one in:

            - indicator (i.e., discrete values);
            - probabilities (i.e., continuous values which sum up to 1);
            - reversed (i.e., probabilities used to compute reserved crossentropy);
            - symmetric (i.e., probabilities used to compute symmetric crossentropy).

        :param weights:
            Whether or not to use sample weights.
        """
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
                        if num_classes == 2:
                            numeric_variables = np.random.uniform(0.001, 0.999, size=NUM_KEYS)
                            if kind in ['reversed', 'symmetric']:
                                model_variables = model.addVars(NUM_KEYS, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='y')
                                model_variables = np.array(model_variables.values())
                                model_assignments = np.random.uniform(0.001, 0.999, size=NUM_KEYS)
                                constraints = (v == p for v, p in zip(model_variables, model_assignments))
                            else:
                                model_variables = model.addVars(NUM_KEYS, vtype=GRB.BINARY, name='y')
                                model_variables = np.array(model_variables.values())
                                model_assignments = np.random.choice([0, 1], size=NUM_KEYS)
                                constraints = (v == p for v, p in zip(model_variables, model_assignments))
                                if kind == 'indicator':
                                    # handle integer predictions in case of hamming distance
                                    numeric_variables = np.round(numeric_variables).astype(int)
                                elif kind != 'probability':
                                    raise ValueError(f"unsupported kind '{kind}'")
                        elif isinstance(num_classes, int) and num_classes > 2:
                            numeric_variables = np.random.uniform(0.001, 0.999, size=(NUM_KEYS, num_classes))
                            numeric_variables = numeric_variables.transpose() / numeric_variables.sum(axis=1)
                            numeric_variables = numeric_variables.transpose()
                            if kind in ['reversed', 'symmetric']:
                                model_variables = np.array(model.addVars(NUM_KEYS, num_classes, vtype=GRB.CONTINUOUS,
                                                                         lb=0, ub=1, name='y').values())
                                model_assignments = np.random.uniform(0.001, 0.999, size=(NUM_KEYS, num_classes))
                                model_assignments = model_assignments.transpose() / model_assignments.sum(axis=1)
                                model_assignments = model_assignments.transpose()
                                constraints = (v == p for v, p in zip(model_variables, model_assignments.flatten()))
                            else:
                                model_variables = model.addVars(NUM_KEYS, num_classes, vtype=GRB.BINARY, name='y')
                                model_variables = np.array(model_variables.values())
                                model_assignments = np.random.choice(range(num_classes), size=NUM_KEYS)
                                model_assignments = one_hot(model_assignments, num_classes=num_classes).astype(int)
                                constraints = (v == p for v, p in zip(model_variables, model_assignments.flatten()))
                                if kind == 'indicator':
                                    # handle integer predictions in case of hamming distance
                                    model_assignments = model_assignments.argmax(axis=1)
                                    numeric_variables = numeric_variables.argmax(axis=1)
                                elif kind != 'probability':
                                    raise ValueError(f"'{kind}' is not a valid kind")
                            model_variables = model_variables.reshape(NUM_KEYS, num_classes)
                        else:
                            raise ValueError(f"'num_classes' should be either None or a tuple, but it is {num_classes}")
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
                    self.assertAlmostEqual(gurobi_objective, scikit_objective, places=PLACES)
