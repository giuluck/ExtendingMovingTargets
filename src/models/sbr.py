"""Semantic-Based Regularized MLP Model."""

from typing import Optional, Callable

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

from moving_targets.util.typing import Matrix, Vector
from src.models.mlp import MLP


def hard_tanh(x, factor=10 ** 6):
    """Approximated sign function using the hyperbolic tangent.

    Args:
        x: the input vector.
        factor: the vector scaling factor.

    Returns:
        An approximated sign function.
    """
    return k.tanh(factor * x)


class SBRBatchGenerator(Sequence):
    """A Batch Generator to be used in Keras model's training.

    Args:
        x: the matrix/dataframe of augmented training samples.
        y: the vector of augmented training labels.
        ground_indices: the ground index of each augmented sample.
        monotonicities: the monotonicity of each augmented sample wrt its ground index.
        batch_size: the batch size.
    """

    def __init__(self, x: Matrix, y: Vector, ground_indices: pd.Series, monotonicities: pd.Series, batch_size: int):
        super(SBRBatchGenerator, self).__init__()
        # compute number of samples in each group
        counts = np.array(ground_indices.value_counts())
        assert np.allclose(counts, counts[0]), "All the ground samples must have the same number of augmented ones."
        self.num_samples = counts[0]
        # get a copy of the whole data
        data = x.copy()
        data['label'] = y.copy()
        data['index'] = ground_indices.copy()
        data['monotonicity'] = monotonicities.copy()
        # place labelled values at the beginning, then shuffle the indices and create batches based on the index value
        data = data.sort_values(['index', 'label'], ascending=[True, False], ignore_index=True).astype('float32')
        shuffle = np.random.permutation(np.unique(ground_indices))
        shuffle = {i: m for i, m in enumerate(shuffle)}
        data['batch'] = data['index'].map(shuffle) // batch_size
        # store list of batches by grouping dataframe by batches
        self.batches = [b.drop(['batch', 'index'], axis=1).reset_index(drop=True) for _, b in data.groupby('batch')]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        # x = first n columns, labels = second to last column, monotonicities = last column
        batch = self.batches[index]
        num_features = len(batch.columns)
        tensor = batch.values.reshape(-1, self.num_samples, num_features).transpose([1, 0, 2])
        return tensor[:, :, :-2].reshape(-1, num_features - 2), (tensor[:, :, -2], tensor[:, :, -1])


class SBR(MLP):
    """A Semantic-Based Regularized MLP architecture for monotonicity shape constraints built with Keras.

    Args:
        alpha: the alpha value for balancing compiled and regularized loss. If None, this is iteratively modified in
               order to be effective at most via Lagrangian dual techniques.
        regularizer_act: the regularizer activation function.
        **kwargs: super-class arguments.
    """

    def __init__(self, alpha: Optional[float] = None, regularizer_act: Optional[Callable] = None, **kwargs):
        super(SBR, self).__init__(**kwargs)
        if alpha is None:
            self.alpha = tf.Variable(0., name='alpha')
            self.alpha_optimizer = Adam()
        else:
            self.alpha = alpha
            self.alpha_optimizer = None
        self.regularizer_act = regularizer_act
        self.alpha_tracker = Mean(name='alpha')
        self.tot_loss_tracker = Mean(name='tot_loss')
        self.def_loss_tracker = Mean(name='def_loss')
        self.reg_loss_tracker = Mean(name='reg_loss')
        self.test_loss_tracker = Mean(name='test_loss')

    def _custom_loss(self, x, y, sign=1):
        labels, monotonicities = y
        pred = self(x, training=True)
        pred = tf.reshape(pred, tf.shape(labels))
        # compiled loss on labeled data, if label is present (the if-else handles the case of unlabeled data only)
        mask = tf.math.logical_not(tf.math.is_nan(labels[0]))
        def_loss = tf.cond(
            pred=tf.math.reduce_any(mask),
            true_fn=lambda: self.compiled_loss(labels[0][mask], pred[0][mask]),
            false_fn=lambda: tf.constant(0.)
        )
        # regularization term computed
        deltas = pred - pred[0]
        if self.regularizer_act is not None:
            deltas = self.regularizer_act(deltas)
        reg_loss = k.mean(k.maximum(0., -monotonicities * deltas))
        # final losses
        return sign * (def_loss + self.alpha * reg_loss), def_loss, reg_loss

    # noinspection PyMissingOrEmptyDocstring
    def train_step(self, d):
        # unpack training data
        x, y = d
        # split trainable variables
        nn_vars = self.trainable_variables[:-1]
        alpha_var = self.trainable_variables[-1:]
        # first optimization step (network parameters)
        with tf.GradientTape() as tape:
            tot_loss, def_loss, reg_loss = self._custom_loss(x, y)
        grads = tape.gradient(tot_loss, nn_vars)
        self.optimizer.apply_gradients(zip(grads, nn_vars))
        # second optimization step (alpha: maximization)
        if self.alpha_optimizer is not None:
            with tf.GradientTape() as tape:
                tot_loss, def_loss, reg_loss = self._custom_loss(x, y, sign=-1)
            grads = tape.gradient(tot_loss, alpha_var)
            self.alpha_optimizer.apply_gradients(zip(grads, alpha_var))
        # loss tracking
        self.alpha_tracker.update_state(self.alpha)
        self.tot_loss_tracker.update_state(abs(tot_loss))
        self.def_loss_tracker.update_state(def_loss)
        self.reg_loss_tracker.update_state(reg_loss)
        return {
            'alpha': self.alpha_tracker.result(),
            'tot_loss': self.tot_loss_tracker.result(),
            'def_loss': self.def_loss_tracker.result(),
            'reg_loss': self.reg_loss_tracker.result()
        }

    def test_step(self, d):
        x, labels = d
        loss = self.compiled_loss(labels, self(x, training=False))
        self.test_loss_tracker.update_state(loss)
        return {
            'loss': self.test_loss_tracker.result()
        }


class UnivariateSBR(SBR):
    """A Semantic-Based Regularized MLP architecture for monotonic univariate functions built with Keras.

    Args:
        direction: the monotonicity direction.
        **kwargs: super-class arguments.
    """

    def __init__(self, direction: int = 1, **kwargs):
        super(UnivariateSBR, self).__init__(**kwargs)
        self.direction = direction

    def _custom_loss(self, x, y, sign=1):
        x = tf.cast(x, tf.float32)
        mask = tf.math.logical_not(tf.math.is_nan(y))
        pred = self(x, training=True)
        # compiled loss on labeled data (the if-else handles the case of unlabeled data only)
        def_loss = tf.cond(
            pred=tf.math.reduce_any(mask),
            true_fn=lambda: self.compiled_loss(y[mask], pred[mask]),
            false_fn=lambda: tf.constant(0.)
        )
        # regularization term computed (sum of violations over each pair of samples)
        x_deltas = tf.repeat(x, tf.size(x), axis=1) - tf.squeeze(x)
        y_deltas = tf.repeat(pred, tf.size(pred), axis=1) - tf.squeeze(pred)
        if self.regularizer_act is not None:
            y_deltas = self.regularizer_act(y_deltas)
        reg_loss = k.sum(k.maximum(0., -self.direction * tf.sign(x_deltas) * y_deltas))
        # final losses
        return sign * (def_loss + self.alpha * reg_loss), def_loss, reg_loss
