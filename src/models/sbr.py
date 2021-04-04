import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

from src.models.mlp import MLP
from src.models.model import Model


def hard_tanh(x, factor=10 ** 6):
    return k.tanh(factor * x)


class SBRBatchGenerator(Sequence):
    def __init__(self, x, y, ground_indices, monotonicities, batch_size):
        super(SBRBatchGenerator, self).__init__()
        # compute number of samples in each group
        self.num_samples = np.sum(ground_indices == 0)
        # get a copy of the whole data
        data = x.copy()
        data['label'] = y
        data['index'] = ground_indices
        data['monotonicity'] = monotonicities
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


class SBR(MLP, Model):
    def __init__(self, output_act=None, h_units=None, scaler=None, alpha=None, regularizer_act=None, input_dim=None):
        super(SBR, self).__init__(output_act=output_act, h_units=h_units, scaler=scaler, input_dim=input_dim)
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
        # compiled loss on labeled data
        def_loss = self.compiled_loss(labels[0], pred[0])
        # regularization term computed
        deltas = pred - pred[0]
        if self.regularizer_act is not None:
            deltas = self.regularizer_act(deltas)
        reg_loss = k.sum(k.maximum(0., -monotonicities * deltas))
        # final losses
        return sign * (def_loss + self.alpha * reg_loss), def_loss, reg_loss

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
