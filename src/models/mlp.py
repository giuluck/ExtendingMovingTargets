"""Basic Multi-Layer Perceptron Model with Keras."""

from typing import Optional, List

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from src.util.preprocessing import Scaler, Scalers


class MLP(Model):
    """A Multi-Layer Perceptron architecture built with Keras."""

    def __init__(self,
                 output_act: Optional[str] = None,
                 h_units: Optional[List[int]] = None,
                 scalers: Scalers = None,
                 input_dim: Optional[int] = None):
        """
        :param output_act:
            The output activation function.

        :param h_units:
            The hidden units.

        :param scalers:
            The x/y scalers to scale input and output data.

        :param input_dim:
            The input dimension.
        """
        super(MLP, self).__init__()

        self.x_scaler: Optional[Scaler] = scalers
        """The input data `Scaler`."""

        self.y_scaler: Optional[Scaler] = None
        """The output data `Scaler`."""

        self.lrs: List[Dense] = [] if h_units is None else [Dense(h, activation='relu') for h in h_units]
        """The list of `tensorflow.keras.layers.Dense` layers."""

        # handle scalers
        if scalers is None:
            self.x_scaler, self.y_scaler = None, None
        elif isinstance(scalers, tuple):
            self.x_scaler, self.y_scaler = scalers

        # handle output layer and tensorflow variables (weights) initialization in case input dimension is explicit
        self.lrs = self.lrs + [Dense(1, activation=output_act)]
        if input_dim is not None:
            self(tf.zeros((1, input_dim)))

    def get_config(self):
        """Overrides Keras method."""
        pass

    def call(self, inputs, training=None, mask=None):
        """Overrides Keras method.

        :param inputs:
            The neural network inputs.

        :param training:
            Overrides Keras parameter.

        :param mask:
            Overrides Keras parameter.
        """
        x = inputs if self.x_scaler is None else self.x_scaler.transform(inputs)
        for layer in self.lrs:
            x = layer(x)
        return x if self.y_scaler is None else self.y_scaler.inverse_transform(x)
