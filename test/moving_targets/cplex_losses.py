import unittest
from typing import Callable, Optional, Tuple

import numpy as np
from docplex.mp.model import Model as CplexModel
from tensorflow.keras.utils import to_categorical as one_hot

from moving_targets.masters import CplexMaster
from test.moving_targets.abstract_losses import TestLosses, SEED, NUM_TESTS, NUM_KEYS, PLACES


class TestCplexLosses(TestLosses, unittest.TestCase):
    """Tests the correctness of Cplex losses."""

    def _losses(self):
        """The solver's losses.

        :returns:
            Cplex's `LossesHandler` object.
        """
        return CplexMaster.losses

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
        # skip reversed and symmetric crossentropy, both binary and categorical
        if TestLosses._tested_loss() in ['reversed_bce', 'reversed_cce', 'symmetric_bce', 'symmetric_cce']:
            return

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
                        # handle integer predictions in case of hamming distance
                        numeric_variables = np.round(numeric_variables).astype(int)
                    elif kind != 'probability':
                        raise ValueError(f"unsupported kind '{kind}'")
                elif isinstance(num_classes, int) and num_classes > 2:
                    model_variables = model.binary_var_matrix(keys1=NUM_KEYS, keys2=num_classes, name='y')
                    model_variables = np.array(list(model_variables.values())).reshape(NUM_KEYS, num_classes)
                    model_assignments = np.random.choice(range(num_classes), size=NUM_KEYS)
                    model_assignments = one_hot(model_assignments, num_classes=num_classes).astype(int)
                    numeric_variables = np.random.uniform(0.001, 0.999, size=(NUM_KEYS, num_classes))
                    numeric_variables = (numeric_variables.transpose() / numeric_variables.sum(axis=1)).transpose()
                    constraints = [v == p for v, p in zip(model_variables.flatten(), model_assignments.flatten())]
                    if kind == 'indicator':
                        # handle integer predictions in case of hamming distance
                        model_assignments = model_assignments.argmax(axis=1)
                        numeric_variables = numeric_variables.argmax(axis=1)
                    elif kind != 'probability':
                        raise ValueError(f"unsupported kind '{kind}'")
                else:
                    raise ValueError(f"'num_classes' should be either None or a tuple, but it is {num_classes}")
            model.add_constraints(constraints)
            model.minimize(model_loss(
                model=model,
                numeric_variables=numeric_variables,
                model_variables=model_variables,
                sample_weight=sample_weight
            ))
            cplex_objective = model.solve().objective_value
            scikit_objective = scikit_loss(model_assignments, numeric_variables, sample_weight=sample_weight)
            self.assertAlmostEqual(cplex_objective, scikit_objective, places=PLACES)
