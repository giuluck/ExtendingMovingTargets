import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

from src.restaurants.models import MLPClassifier


class SBRBatchGenerator(Sequence):
    def __init__(self, x, y, batch_size):
        super(SBRBatchGenerator, self).__init__()
        data = x.reset_index(drop=True).join(y.reset_index(drop=True))
        data = data.sort_values(['index', 'clicked'], ascending=[True, False], ignore_index=True)
        data['batch'] = data.index // (len(data[data['index'] == 0]) * batch_size)
        self.batches = [b.drop(['batch', 'index'], axis=1).reset_index(drop=True) for _, b in data.groupby('batch')]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        batch: pd.DataFrame = self.batches[index]
        x = batch[['avg_rating', 'num_reviews', 'D', 'DD', 'DDD', 'DDDD']]
        y = batch[['clicked', 'monotonicity']]
        return x.values, y.values


class SBRClassifier(MLPClassifier):
    def __init__(self, num_augmented_samples, hidden=None, scaler=None, alpha=None, reg_activation=None):
        super(SBRClassifier, self).__init__(hidden=hidden, scaler=scaler)
        self.num_augmented_samples = num_augmented_samples
        if alpha is None:
            self.alpha = tf.Variable(0., name='alpha')
            self.alpha_optimizer = Adam()
        else:
            self.alpha = alpha
            self.optimize_alpha = None
        self.reg_activation = reg_activation
        self.alpha_tracker = Mean(name='alpha')
        self.tot_loss_tracker = Mean(name='tot_loss')
        self.def_loss_tracker = Mean(name='def_loss')
        self.reg_loss_tracker = Mean(name='reg_loss')
        self.test_loss_tracker = Mean(name='test_loss')

    def __custom_loss(self, x, labels, monotonicities, sign=1):
        # predicted click-through-rate
        ctr = self(x, training=True)
        # compiled loss (binary crossentropy) on labeled data
        def_loss = self.compiled_loss(labels[labels != -1], ctr[labels != -1])
        # regularization term computed
        original_samples_ctr = tf.reshape(ctr, (-1, self.num_augmented_samples))[:, 0]
        original_samples_ctr = tf.repeat(original_samples_ctr, self.num_augmented_samples)
        deltas = tf.reshape(ctr, (-1,)) - original_samples_ctr
        if self.reg_activation is not None:
            deltas = self.reg_activation(deltas)
        reg_loss = k.sum(k.maximum(0., -monotonicities * deltas))
        # final losses
        return sign * (def_loss + self.alpha * reg_loss), def_loss, reg_loss

    def train_step(self, d):
        # unpack training data
        x, y = d
        labels = y[:, 0:1]
        monotonicities = y[:, 1:2]
        # split trainable variables
        nn_vars = self.trainable_variables[:-1]
        alpha_var = self.trainable_variables[-1:]
        # first optimization step (network parameters)
        with tf.GradientTape() as tape:
            tot_loss, def_loss, reg_loss = self.__custom_loss(x, labels, monotonicities)
        grads = tape.gradient(tot_loss, nn_vars)
        self.optimizer.apply_gradients(zip(grads, nn_vars))
        # second optimization step (alpha: maximization)
        if self.alpha_optimizer is not None:
            with tf.GradientTape() as tape:
                tot_loss, def_loss, reg_loss = self.__custom_loss(x, labels, monotonicities, sign=-1)
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
