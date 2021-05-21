from typing import Optional, List, Any

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from src.util.preprocessing import Scaler


class MLP(Model):
    def __init__(self, output_act: Optional[str] = None, h_units: Optional[List[int]] = None, scalers: Any = None,
                 input_dim: Optional[int] = None):
        super(MLP, self).__init__()
        # scalers
        self.x_scaler: Optional[Scaler] = scalers
        self.y_scaler: Optional[Scaler] = None
        if scalers is None:
            self.x_scaler, self.y_scaler = None, None
        elif isinstance(scalers, tuple):
            self.x_scaler, self.y_scaler = scalers
        # layers
        self.lrs: List[Dense] = [] if h_units is None else [Dense(h, activation='relu') for h in h_units]
        self.lrs = self.lrs + [Dense(1, activation=output_act)]
        # handles tensorflow variables (weights) initialization in case input dimension is explicit
        if input_dim is not None:
            self(tf.zeros((1, input_dim)))

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        x = inputs if self.x_scaler is None else self.x_scaler.transform(inputs)
        for layer in self.lrs:
            x = layer(x)
        return x if self.y_scaler is None else self.y_scaler.inverse_transform(x)
