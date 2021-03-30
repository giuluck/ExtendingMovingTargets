from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model as KerasModel

from src.restaurants.models import Model as RestaurantsModel


class MLP(RestaurantsModel, KerasModel):
    def __init__(self, output_act, h_units=None, scaler=None):
        super(MLP, self).__init__()
        self.scaler = scaler
        self.lrs = [] if h_units is None else [Dense(h, activation='relu') for h in h_units]
        self.lrs = self.lrs + [Dense(1, activation=output_act)]

    def get_config(self):
        pass

    def predict(self, x):
        return super(RestaurantsModel, self).predict(x)

    def call(self, inputs, training=None, mask=None):
        x = inputs if self.scaler is None else self.scaler.transform(inputs)
        for layer in self.lrs:
            x = layer(x)
        return x
