"""Semantic-Based Regularized MLP Model."""

from typing import Optional, Callable, Union, Tuple, List, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

from moving_targets.util.typing import Matrix, Vector
from src.models.mlp import MLP
from src.util.preprocessing import Scalers


def hard_tanh(x, factor=10 ** 6):
    """Approximated sign function using the hyperbolic tangent.

    :param x:
        The input vector.

    :param factor:
        The scaling factor.

    :return:
        An approximated sign function.
    """
    return k.tanh(factor * x)


class SBRBatchGenerator(Sequence):
    """A Batch Generator to be used in Keras model's training."""

    def __init__(self, x: Matrix, y: Vector, ground_indices: pd.Series, monotonicities: pd.Series, batch_size: int):
        """
        :param x:
            The matrix/dataframe of augmented training samples.

        :param y:
            The vector of augmented training labels.

        :param ground_indices:
            The ground index of each augmented sample.

        :param monotonicities:
            The monotonicity of each augmented sample wrt its ground index.

        :param batch_size: the batch size.
        """
        super(SBRBatchGenerator, self).__init__()
        # compute number of samples in each group
        counts = np.array(ground_indices.value_counts())
        assert np.allclose(counts, counts[0]), "All the ground samples must have the same number of augmented ones."

        self.num_samples: int = counts[0]
        """The number of samples in each batch."""

        self.batches = None
        """The inner list of collected batches."""

        # get a copy of the whole data (and map indices from random range into range [0, num_grounds]
        data = x.copy()
        data['label'] = y.copy()
        data['index'] = ground_indices.copy()
        data['index'] = data['index'].map({old: new for new, old in enumerate(ground_indices.unique())})
        data['monotonicity'] = monotonicities.copy()
        # place labelled values at the beginning, then shuffle the indices and create batches based on the index value
        data = data.sort_values(['index', 'label'], ascending=[True, False], ignore_index=True).astype('float32')
        shuffle = np.random.permutation(data['index'].unique())
        data['batch'] = data['index'].map({i: m for i, m in enumerate(shuffle)}) // batch_size

        # store list of batches by grouping dataframe by batches
        self.batches = [b.drop(['batch', 'index'], axis=1).reset_index(drop=True) for _, b in data.groupby('batch')]

    def __len__(self) -> int:
        """Necessary method for the Batch Generator.

        :return:
            The number of batches.
        """
        return len(self.batches)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Necessary method for the Batch Generator.

        :param index:
            The batch index.

        :return:
            The index-th batch in the form of (<training_data>, (<ground_truths>, <monotonicities>)).
        """
        # x = first n columns, labels = second to last column, monotonicities = last column
        batch = self.batches[index]
        num_features = len(batch.columns)
        tensor = batch.values.reshape(-1, self.num_samples, num_features).transpose([1, 0, 2])
        return tensor[:, :, :-2].reshape(-1, num_features - 2), (tensor[:, :, -2], tensor[:, :, -1])


class SBR(MLP):
    """A Semantic-Based Regularized MLP architecture for monotonicity shape constraints built with Keras."""

    def __init__(self,
                 output_act: Optional[str] = None,
                 h_units: Optional[List[int]] = None,
                 scalers: Scalers = None,
                 input_dim: Optional[int] = None,
                 alpha: Union[None, float, OptimizerV2] = None,
                 regularizer_act: Optional[Callable] = None):
        super(SBR, self).__init__(output_act=output_act, h_units=h_units, scalers=scalers, input_dim=input_dim)
        """
        :param output_act:
            The output activation function.
        
        :param h_units:
            The hidden units.
        
        :param scalers:
            The x/y scalers to scale input and output data.
        
        :param input_dim:
            The input dimension.
        
        :param alpha:
            The alpha value for balancing compiled and regularized loss.
            
            If None, this is iteratively modified in order to be effective at most via Lagrangian dual techniques.
        
        :param regularizer_act:
            The regularizer activation function.
        """
        self.alpha = None
        """The alpha value for balancing compiled and regularized loss."""

        self.alpha_optimizer = None
        """The optimizer to iteratively modify alpha in order to be effective at most via Lagrangian dual techniques."""

        self.regularizer_act = regularizer_act
        """The regularizer activation function."""

        self._alpha_tracker = Mean(name='alpha')
        """The tracker of alpha values during the training process."""

        self._tot_loss_tracker = Mean(name='tot_loss')
        """The tracker of total loss values during the training process."""

        self._def_loss_tracker = Mean(name='def_loss')
        """The tracker of default loss values during the training process."""

        self._reg_loss_tracker = Mean(name='reg_loss')
        """The tracker of regularizer loss values during the training process."""

        self._test_loss_tracker = Mean(name='test_loss')
        """The tracker of test loss values during the training process."""

        # we use a tf.Variable for alpha in order to retrieve nn_vars in an easier way as trainable_variables[:-1]
        if isinstance(alpha, float):
            self.alpha = tf.Variable(alpha, name='alpha')
            self.alpha_optimizer = None
        else:
            self.alpha = tf.Variable(0., name='alpha')
            self.alpha_optimizer = Adam(learning_rate=1.0) if alpha is None else alpha

    def _custom_loss(self, x: tf.Tensor, y: Tuple[tf.Tensor, tf.Tensor], sign: int = 1) -> Tuple[float, float, float]:
        """Computes the custom losses.

        :param x:
            The input data.

        :param y:
            The tuple (<ground_truths>, <monotonicities>).

        :param sign:
            Whether to minimize the loss (1) or to maximize it (-1) depending on the training step.

        :return:
            A tuple of the form (<total_loss>, <default_loss>, <regularizer_loss>), where the total loss is computed as:
            <total_loss> = <sign> * (<default_loss> + <alpha> * <regularizer_loss>)
        """
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

    def train_step(self, d: Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]) -> Dict[str, float]:
        """Overrides keras `train_step` method.

        :param d:
            The batch, having the form (<training_data>, (<ground_truths>, <monotonicities>)).

        :return:
            A dictionary containing the values of <alpha>, <total_loss>, <default_loss>, and <regularization_loss>.
        """
        # unpack training data
        x, y = d
        nn_vars = self.trainable_variables[:-1]
        alpha_var = self.trainable_variables[-1:]
        # first optimization step: network parameters with alpha (last var) excluded -> loss minimization
        with tf.GradientTape() as tape:
            tot_loss, def_loss, reg_loss = self._custom_loss(x, y, sign=1)
        grads = tape.gradient(tot_loss, nn_vars)
        self.optimizer.apply_gradients(zip(grads, nn_vars))
        # second optimization step: alpha only -> loss maximization
        if self.alpha_optimizer is not None:
            with tf.GradientTape() as tape:
                tot_loss, def_loss, reg_loss = self._custom_loss(x, y, sign=-1)
            grads = tape.gradient(tot_loss, alpha_var)
            self.alpha_optimizer.apply_gradients(zip(grads, alpha_var))
        # loss tracking
        self._alpha_tracker.update_state(self.alpha)
        self._tot_loss_tracker.update_state(abs(tot_loss))
        self._def_loss_tracker.update_state(def_loss)
        self._reg_loss_tracker.update_state(reg_loss)
        return {
            'alpha': self._alpha_tracker.result(),
            'tot_loss': self._tot_loss_tracker.result(),
            'def_loss': self._def_loss_tracker.result(),
            'reg_loss': self._reg_loss_tracker.result()
        }

    def test_step(self, d: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, float]:
        """Overrides keras `test_step` method.

        :param d:
            The batch, having the form (<input_data>, <ground_truths>).

        :return:
            A dictionary containing the values of <test_loss>.
        """
        x, labels = d
        loss = self.compiled_loss(labels, self(x, training=False))
        self._test_loss_tracker.update_state(loss)
        return {
            'loss': self._test_loss_tracker.result()
        }


class UnivariateSBR(SBR):
    """A Semantic-Based Regularized MLP architecture for monotonic univariate functions built with Keras."""

    def __init__(self,
                 output_act: Optional[str] = None,
                 h_units: Optional[List[int]] = None,
                 scalers: Scalers = None,
                 input_dim: Optional[int] = None,
                 alpha: Union[None, float, OptimizerV2] = None,
                 regularizer_act: Optional[Callable] = None,
                 direction: int = 1):
        """
        :param output_act:
            The output activation function.

        :param h_units:
            The hidden units.

        :param scalers:
            The x/y scalers to scale input and output data.

        :param input_dim:
            The input dimension.

        :param alpha:
            The alpha value for balancing compiled and regularized loss.

            If None, this is iteratively modified in order to be effective at most via Lagrangian dual techniques.

        :param regularizer_act:
            The regularizer activation function.

        :param direction:
            The monotonicity direction.
        """
        super(UnivariateSBR, self).__init__(output_act=output_act, h_units=h_units, scalers=scalers,
                                            input_dim=input_dim, alpha=alpha, regularizer_act=regularizer_act)
        self.direction = direction
        """The monotonicity direction."""

    def _custom_loss(self, x: tf.Tensor, y: Tuple[tf.Tensor, tf.Tensor], sign: int = 1) -> Tuple[float, float, float]:
        """Computes the custom losses.

        :param x:
            The input data.

        :param y:
            The tuple (<ground_truths>, <monotonicities>).

        :param sign:
            Whether to minimize the loss (1) or to maximize it (-1) depending on the training step.

        :return:
            A tuple of the form (<total_loss>, <default_loss>, <regularizer_loss>), where the total loss is computed as:
            <total_loss> = <sign> * (<default_loss> + <alpha> * <regularizer_loss>)
        """
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
        reg_loss = k.mean(k.maximum(0., -self.direction * tf.sign(x_deltas) * y_deltas))
        # final losses
        return sign * (def_loss + self.alpha * reg_loss), def_loss, reg_loss
