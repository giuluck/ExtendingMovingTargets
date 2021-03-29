from src.moving_targets.learners import Learner
from src.restaurants.models import MLP


class MTLearner(Learner):
    def __init__(self, h_units=None, scaler=None, restart_fit=True, loss='mse', optimizer='adam', **kwargs):
        super(MTLearner, self).__init__()
        self.h_units = h_units
        self.scaler = scaler
        self.loss = loss
        self.optimizer = optimizer
        self.restart_fit = restart_fit
        self.fit_args = kwargs
        self.model = MLP(output_act='sigmoid', h_units=self.h_units, scaler=self.scaler)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self, macs, x, y, iteration):
        if self.restart_fit:
            self.model = MLP(output_act='sigmoid', h_units=self.h_units, scaler=self.scaler)
            self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.model.fit(x[y != -1], y[y != -1], **self.fit_args)

    def predict(self, x):
        return self.predict_proba(x)

    def predict_proba(self, x):
        return self.model.predict(x).reshape(-1, )
