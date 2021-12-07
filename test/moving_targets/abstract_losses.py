"""Losses Tests."""
import inspect
import unittest
from typing import Callable, Optional, Tuple, List

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as k
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, precision_score
from tensorflow.keras.utils import to_categorical as one_hot

from moving_targets.learners.learner import Classifier
from moving_targets.masters import LossesHandler

SEED: int = 0
"""The chosen random seed."""

NUM_KEYS: int = 10
"""The number of features, i.e., size of the vector."""

NUM_TESTS: int = 10
"""The number of tests carried out for the same loss."""

PLACES: int = 3
"""The number of digits passed to the `assertAlmostEqual()` method."""

k.set_epsilon(1e-15)


class TestLosses(unittest.TestCase):
    """Template class to test the correctness of the implemented losses respectively to each backend solver."""

    @staticmethod
    def _custom_loss(name: str) -> Callable:
        """Builds a custom loss callable function.


        :param name:
            A loss name in ['sae', 'sse', 'binary hamming', 'categorical hamming', 'reversed binary crossentropy',
            'reversed categorical crossentropy', 'symmetric binary crossentropy', 'symmetric categorical crossentropy'].

        :return:
            A custom loss obtained either from scikit learn losses or from tensorflow primitives, e.g., SAE (sum of
            squared errors) is obtained as MAE(y, p) * NUM_KEY.
        """

        def _ce_handler(y: np.ndarray, p: np.ndarray, sw: Optional[np.ndarray], loss_fn: Callable) -> np.ndarray:
            """Computes the categorical crossentropy using tensorflow primitives.

            :param y:
                The ground truths.

            :param p:
                The computed predictions.

            :param sw:
                The (optional) sample weights.

            :param loss_fn:
                The kind of crossentropy (reversed or symmetric), handled via a callable function.

            :return:
                A numpy scalar representing the loss value.
            """
            y = tf.cast(y, tf.float32)
            p = tf.cast(p, tf.float32)
            sw = tf.cast(tf.constant(1.0) if sw is None else len(sw) * tf.constant(sw) / k.sum(sw), tf.float32)
            return k.mean(sw * loss_fn(y, p)).numpy()

        if name == 'sae':
            return lambda y, p, sample_weight: NUM_KEYS * mean_absolute_error(y, p, sample_weight=sample_weight)
        elif name == 'sse':
            return lambda y, p, sample_weight: NUM_KEYS * mean_squared_error(y, p, sample_weight=sample_weight)
        elif name in ['binary hamming', 'categorical hamming']:
            return lambda y, p, sample_weight: 1 - precision_score(y, p, sample_weight=sample_weight, average='micro')
        elif name in ['reversed binary crossentropy', 'reversed categorical crossentropy']:
            fn = k.binary_crossentropy if 'binary' in name else k.categorical_crossentropy
            return lambda y, p, sample_weight: _ce_handler(y, p, sample_weight, lambda yy, pp: fn(pp, yy))
        elif name in ['symmetric binary crossentropy', 'symmetric categorical crossentropy']:
            fn = k.binary_crossentropy if 'binary' in name else k.categorical_crossentropy
            return lambda y, p, sample_weight: _ce_handler(y, p, sample_weight, lambda yy, pp: fn(yy, pp) + fn(pp, yy))
        else:
            raise ValueError(f'{name} is not a valid loss')

    def _losses(self) -> LossesHandler:
        """The solver's losses.

        :return:
            Cvxpy's `LossesHandler` object.
        """
        raise NotImplementedError(f"Please implement abstract method '_losses'")

    def _unsupported(self) -> List[str]:
        """The list of unsupported losses.

        :return:
            A list of strings representing the unsupported losses.
        """
        raise NotImplementedError(f"Please implement abstract method '_unsupported'")

    def _objective(self, values: np.ndarray, classes: Optional[Tuple[int, str]], loss: Callable) -> float:
        """Creates the model and computes the objective given the loss.

        :param values:
            The numeric assignments (i.e., ground values).

        :param classes:
            Either None in case of regression or the number of output classes.

        :param loss:
            The loss function.

        :return:
            The objective value.
        """
        raise NotImplementedError(f"Please implement abstract method '_objective'")

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
            try:
                # generate random data (ground truths, predictions, and sample weights) and fix the model variable to be
                # the same as the ground truths in order to obtain the loss as objective function from the minimization
                sample_weight = np.random.uniform(size=NUM_KEYS) if weights else None
                if classes is None:
                    numeric_variables = np.random.uniform(-100, 100, size=NUM_KEYS)
                    model_assignments = np.random.uniform(-100, 100, size=NUM_KEYS)
                else:
                    num_classes, kind = classes
                    if num_classes == 2:
                        numeric_variables = np.random.uniform(0.001, 0.999, size=NUM_KEYS)
                        if kind in ['reversed', 'symmetric']:
                            model_assignments = np.random.uniform(0.001, 0.999, size=NUM_KEYS)
                        elif kind in ['probability', 'indicator']:
                            model_assignments = np.random.choice([0, 1], size=NUM_KEYS)
                        else:
                            raise ValueError(f"unsupported kind '{kind}'")
                    elif isinstance(num_classes, int) and num_classes > 2:
                        numeric_variables = np.random.uniform(0.001, 0.999, size=(NUM_KEYS, num_classes))
                        numeric_variables = numeric_variables.transpose() / numeric_variables.sum(axis=1)
                        numeric_variables = numeric_variables.transpose()
                        if kind in ['reversed', 'symmetric']:
                            model_assignments = np.random.uniform(0.001, 0.999, size=(NUM_KEYS, num_classes))
                            model_assignments = model_assignments.transpose() / model_assignments.sum(axis=1)
                            model_assignments = model_assignments.transpose()
                        elif kind in ['probability', 'indicator']:
                            model_assignments = np.random.choice(range(num_classes), size=NUM_KEYS)
                            model_assignments = one_hot(model_assignments, num_classes=num_classes).astype(int)
                        else:
                            raise ValueError(f"'{kind}' is not a valid kind")
                    else:
                        raise ValueError(f"'classes' should be either None or a tuple, but it is {num_classes}")
                if classes is not None and classes[1] == 'indicator':
                    numeric_variables = Classifier.get_classes(numeric_variables)
                    model_objective = self._objective(
                        values=model_assignments,
                        classes=classes,
                        loss=lambda mdl, var: model_loss(mdl, numeric_variables, var, sample_weight)
                    )
                    model_assignments = Classifier.get_classes(model_assignments)
                else:
                    model_objective = self._objective(
                        values=model_assignments,
                        classes=classes,
                        loss=lambda mdl, var: model_loss(mdl, numeric_variables, var, sample_weight)
                    )
                scikit_objective = scikit_loss(model_assignments, numeric_variables, sample_weight=sample_weight)
                self.assertAlmostEqual(model_objective, scikit_objective, places=PLACES)
            except ValueError as exception:
                tested_loss = inspect.stack()[1][3].replace('test_', '').replace('_weights', '')
                self.assertIn(tested_loss, self._unsupported())
                self.assertTrue(str(exception).startswith('This solver cannot deal with'))

    def test_sae(self):
        """Tests Sum of Absolute Errors loss."""
        self._test(
            model_loss=self._losses().sum_of_absolute_errors,
            scikit_loss=TestLosses._custom_loss('sae'),
            classes=None,
            weights=False
        )

    def test_sae_weights(self):
        """Tests Sum of Absolute Errors loss with Sample Weights."""
        self._test(
            model_loss=self._losses().sum_of_absolute_errors,
            scikit_loss=TestLosses._custom_loss('sae'),
            classes=None,
            weights=True
        )

    def test_sse(self):
        """Tests Sum of Squared Errors loss."""
        self._test(
            model_loss=self._losses().sum_of_squared_errors,
            scikit_loss=TestLosses._custom_loss('sse'),
            classes=None,
            weights=False
        )

    def test_sse_weights(self):
        """Tests Sum of Squared Errors loss with Sample Weights."""
        self._test(
            model_loss=self._losses().sum_of_squared_errors,
            scikit_loss=TestLosses._custom_loss('sse'),
            classes=None,
            weights=True
        )

    def test_mae(self):
        """Tests Mean Absolute Error loss."""
        self._test(
            model_loss=self._losses().mean_absolute_error,
            scikit_loss=mean_absolute_error,
            classes=None,
            weights=False
        )

    def test_mae_weights(self):
        """Tests Mean Absolute Error loss with Sample Weights."""
        self._test(
            model_loss=self._losses().mean_absolute_error,
            scikit_loss=mean_absolute_error,
            classes=None,
            weights=True
        )

    def test_mse(self):
        """Tests Mean Squared Error loss with Sample Weights."""
        self._test(
            model_loss=self._losses().mean_squared_error,
            scikit_loss=mean_squared_error,
            classes=None,
            weights=False
        )

    def test_mse_weights(self):
        """Tests Mean Squared Error loss with Sample Weights."""
        self._test(
            model_loss=self._losses().mean_squared_error,
            scikit_loss=mean_squared_error,
            classes=None,
            weights=True
        )

    def test_bh(self):
        """Tests Binary Hamming distance."""
        self._test(
            model_loss=self._losses().binary_hamming,
            scikit_loss=TestLosses._custom_loss('binary hamming'),
            classes=(2, 'indicator'),
            weights=False
        )

    def test_bh_weights(self):
        """Tests Binary Hamming distance with Sample Weights."""
        self._test(
            model_loss=self._losses().binary_hamming,
            scikit_loss=TestLosses._custom_loss('binary hamming'),
            classes=(2, 'indicator'),
            weights=True)

    def test_bce(self):
        """Tests Binary Crossentropy loss."""
        self._test(
            model_loss=self._losses().binary_crossentropy,
            scikit_loss=log_loss,
            classes=(2, 'probability'),
            weights=False
        )

    def test_bce_weights(self):
        """Tests Binary Crossentropy loss with Sample Weights."""
        self._test(
            model_loss=self._losses().binary_crossentropy,
            scikit_loss=log_loss,
            classes=(2, 'probability'),
            weights=True
        )

    def test_reversed_bce(self):
        """Tests Reversed Binary Crossentropy loss."""
        self._test(
            model_loss=self._losses().reversed_binary_crossentropy,
            scikit_loss=TestLosses._custom_loss('reversed binary crossentropy'),
            classes=(2, 'reversed'),
            weights=False
        )

    def test_reversed_bce_weights(self):
        """Tests Reversed Binary Crossentropy loss with Sample Weights."""
        self._test(
            model_loss=self._losses().reversed_binary_crossentropy,
            scikit_loss=TestLosses._custom_loss('reversed binary crossentropy'),
            classes=(2, 'reversed'),
            weights=True
        )

    def test_symmetric_bce(self):
        """Tests Symmetric Binary Crossentropy loss."""
        self._test(
            model_loss=self._losses().symmetric_binary_crossentropy,
            scikit_loss=TestLosses._custom_loss('symmetric binary crossentropy'),
            classes=(2, 'symmetric'),
            weights=False
        )

    def test_symmetric_bce_weights(self):
        """Tests Symmetric Binary Crossentropy loss with Sample Weights."""
        self._test(
            model_loss=self._losses().symmetric_binary_crossentropy,
            scikit_loss=TestLosses._custom_loss('symmetric binary crossentropy'),
            classes=(2, 'symmetric'),
            weights=True
        )

    def test_ch(self):
        """Tests Categorical Hamming distance."""
        self._test(
            model_loss=self._losses().categorical_hamming,
            scikit_loss=TestLosses._custom_loss('categorical hamming'),
            classes=(5, 'indicator'),
            weights=False
        )

    def test_ch_weights(self):
        """Tests Categorical Hamming distance with Sample Weights."""
        self._test(
            model_loss=self._losses().categorical_hamming,
            scikit_loss=TestLosses._custom_loss('categorical hamming'),
            classes=(5, 'indicator'),
            weights=True
        )

    def test_cce(self):
        """Tests Categorical Crossentropy loss."""
        self._test(
            model_loss=self._losses().categorical_crossentropy,
            scikit_loss=log_loss,
            classes=(5, 'probability'),
            weights=False
        )

    def test_cce_weights(self):
        """Tests Categorical Crossentropy loss with Sample Weights."""
        self._test(
            model_loss=self._losses().categorical_crossentropy,
            scikit_loss=log_loss,
            classes=(5, 'probability'),
            weights=True
        )

    def test_reversed_cce(self):
        """Tests Reversed Categorical Crossentropy loss."""
        self._test(
            model_loss=self._losses().reversed_categorical_crossentropy,
            scikit_loss=TestLosses._custom_loss('reversed categorical crossentropy'),
            classes=(5, 'reversed'),
            weights=False
        )

    def test_reversed_cce_weights(self):
        """Tests Reversed Categorical Crossentropy loss with Sample Weights."""
        self._test(
            model_loss=self._losses().reversed_categorical_crossentropy,
            scikit_loss=TestLosses._custom_loss('reversed categorical crossentropy'),
            classes=(5, 'reversed'),
            weights=True
        )

    def test_symmetric_cce(self):
        """Tests Symmetric Categorical Crossentropy loss."""
        self._test(
            model_loss=self._losses().symmetric_categorical_crossentropy,
            scikit_loss=TestLosses._custom_loss('symmetric categorical crossentropy'),
            classes=(5, 'symmetric'),
            weights=False
        )

    def test_symmetric_cce_weights(self):
        """Tests Reversed Categorical Crossentropy loss with Sample Weights."""
        self._test(
            model_loss=self._losses().symmetric_categorical_crossentropy,
            scikit_loss=TestLosses._custom_loss('symmetric categorical crossentropy'),
            classes=(5, 'symmetric'),
            weights=True
        )
