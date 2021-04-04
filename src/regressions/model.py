import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as k
from sklearn.metrics import r2_score

from src.models import SBR, Model
from src.util.preprocessing import Scaler


class UnivariateSBR(SBR):
    def __init__(self,
                 direction=1,
                 output_act=None,
                 h_units=None,
                 scaler=None,
                 alpha=None,
                 regularizer_act=None,
                 input_dim=None):
        super(UnivariateSBR, self).__init__(output_act=output_act,
                                            h_units=h_units,
                                            scaler=scaler,
                                            alpha=alpha,
                                            regularizer_act=regularizer_act,
                                            input_dim=input_dim)
        self.direction = direction

    def _custom_loss(self, x, y, sign=1):
        x = tf.cast(x, tf.float32)
        mask = tf.math.logical_not(tf.math.is_nan(y))
        pred = self(x, training=True)
        # compiled loss on labeled data
        def_loss = self.compiled_loss(y[mask], pred[mask])
        # regularization term computed (sum of violations over each pair of samples)
        monotonicities = -self.direction * tf.sign((tf.repeat(x, tf.size(x), axis=1) - tf.squeeze(x)))
        deltas = tf.repeat(pred, tf.size(pred), axis=1) - tf.squeeze(pred)
        if self.regularizer_act is not None:
            deltas = self.regularizer_act(deltas)
        reg_loss = k.sum(k.maximum(0., monotonicities * deltas))
        # final losses
        return sign * (def_loss + self.alpha * reg_loss), def_loss, reg_loss


def import_extension_methods():
    def cars_summary(model, scalers=None, res=100, xlim=(0, 60), ylim=(0, 120), figsize=(10, 4), **kwargs):
        plt.figure(figsize=figsize)
        x_scaler, y_scaler = (Scaler.get_default(1), Scaler.get_default(1)) if scalers is None else scalers
        # evaluation data
        summary = []
        for title, (x, y) in kwargs.items():
            p = y_scaler.invert(model.predict(x))
            x = x_scaler.invert(x['price'])
            y = y_scaler.invert(y)
            summary.append(f'{r2_score(y, p):.4} ({title} r2)')
            sns.scatterplot(x=x, y=y, alpha=0.25, sizes=0.25, label=title)
        print(', '.join(summary))
        # estimated function
        x = np.linspace(x_scaler.transform(xlim)[0], x_scaler.transform(xlim)[1], res)
        y = model.predict(x).flatten()
        sns.lineplot(x=x_scaler.invert(x), y=y_scaler.invert(y), color='black').set(
            xlabel='price', ylabel='sales', title='Estimated Function'
        )
        plt.xlim(xlim)
        plt.ylim(ylim)

    def puzzles_summary(model, scalers=None, res=30, figsize=(10, 4), tight_layout=True, **kwargs):
        features = ['word_count', 'star_rating', 'num_reviews']
        fig, axes = plt.subplots(1, 3, sharey='all', tight_layout=tight_layout, figsize=figsize)
        x_scaler, y_scaler = (Scaler.get_default(1), Scaler.get_default(1)) if scalers is None else scalers
        # evaluation data
        summary = []
        for title, (x, y) in kwargs.items():
            p = y_scaler.invert(model.predict(x))
            x = x_scaler.invert(x)
            y = y_scaler.invert(y)
            summary.append(f'{r2_score(y, p):.4} ({title} r2)')
            for ax, feat in zip(axes, features):
                sns.scatterplot(x=x[feat], y=y, alpha=0.25, sizes=0.25, label=title, ax=ax)
        print(', '.join(summary))
        # estimated functions
        grid = np.meshgrid(np.linspace(0, 1, res), np.linspace(0, 1, res), np.linspace(0, 1, res))
        grid = np.concatenate([x.reshape(-1, 1) for x in grid], axis=1)
        grid = pd.DataFrame(grid, columns=features)
        pred = y_scaler.invert(model.predict(grid)).flatten()
        grid = x_scaler.invert(grid)
        for ax, feat in zip(axes, features):
            sns.lineplot(x=grid[feat], y=pred, color='black', ci=99, ax=ax).set(
                xlabel=feat, ylabel='label', xlim=(grid[feat].min(), grid[feat].max()), ylim=(pred.min(), pred.max())
            )
        fig.suptitle('Estimated Functions')

    Model.cars_summary = cars_summary
    Model.puzzles_summary = puzzles_summary
