import unittest
from typing import Callable, Optional, Tuple

import cvxpy as cp
import numpy as np

from moving_targets.masters import LossesHandler
from moving_targets.masters.cvxpy_master import CvxpyMaster
from test.moving_targets.abstract_losses import TestLosses, SEED, NUM_TESTS, NUM_KEYS, PLACES


class TestCvxpyLosses(TestLosses, unittest.TestCase):
    """Tests the correctness of Cvxpy losses."""

    def _losses(self) -> LossesHandler:
        """The solver's losses.

        :returns:
            Cvxpy's `LossesHandler` object.
        """
        return CvxpyMaster.losses

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
        # skip hamming distance, crossentropy, and symmetric binary, both binary and categorical
        if TestLosses._tested_loss() in ['bh', 'ch', 'bce', 'cce', 'symmetric_bce', 'symmetric_cce']:
            return
        # fix a random seed for data generation and repeat for the given number of tests
        np.random.seed(SEED)
        for i in range(NUM_TESTS):
            # generate random data (ground truths, predictions, and sample weights) and fix the model variable to be
            # the same as the ground truths in order to obtain the loss as objective function from the minimization
            sample_weight = np.random.uniform(size=NUM_KEYS) if weights else None
            if classes is None:
                model_variables = [cp.Variable((1,)) for _ in range(NUM_KEYS)]
                model_assignments = np.random.uniform(-100, 100, size=NUM_KEYS)
                numeric_variables = np.random.uniform(-100, 100, size=NUM_KEYS)
                constraints = []
                for v, p in zip(model_variables, model_assignments):
                    v.lb, v.ub = -100, 100
                    constraints.append(v == p)
            else:
                num_classes, kind = classes
                if num_classes == 2:
                    model_variables = [cp.Variable((1,)) for _ in range(NUM_KEYS)]
                    numeric_variables = np.random.uniform(0.001, 0.999, size=NUM_KEYS)
                    model_assignments = np.random.uniform(0.001, 0.999, size=NUM_KEYS)
                    constraints = []
                    for v, p in zip(model_variables, model_assignments):
                        v.lb, v.ub = 0, 1
                        constraints.append(v == p)
                elif isinstance(num_classes, int) and num_classes > 2:
                    model_variables = [cp.Variable((1,)) for _ in range(NUM_KEYS * num_classes)]
                    numeric_variables = np.random.uniform(0.001, 0.999, size=(NUM_KEYS, num_classes))
                    numeric_variables = numeric_variables.transpose() / numeric_variables.sum(axis=1)
                    numeric_variables = numeric_variables.transpose()
                    model_assignments = np.random.uniform(0.001, 0.999, size=(NUM_KEYS, num_classes))
                    model_assignments = model_assignments.transpose() / model_assignments.sum(axis=1)
                    model_assignments = model_assignments.transpose()
                    constraints = []
                    for v, p in zip(model_variables, model_assignments.flatten()):
                        v.lb, v.ub = 0, 1
                        constraints.append(v == p)
                    model_variables = np.array(model_variables).reshape(NUM_KEYS, num_classes)
                else:
                    raise ValueError(f"'num_classes' should be either None or a tuple, but it is {num_classes}")
            model_variables = np.array(model_variables)
            objective = cp.Minimize(model_loss(
                model=constraints,
                numeric_variables=numeric_variables,
                model_variables=model_variables,
                sample_weight=sample_weight
            ))
            model = cp.Problem(objective, constraints)
            cvxpy_objective = model.solve()
            scikit_objective = scikit_loss(model_assignments, numeric_variables, sample_weight=sample_weight)
            self.assertAlmostEqual(cvxpy_objective, scikit_objective, places=PLACES)
