import numpy as np
from sklearn.metrics import roc_auc_score, r2_score

from src import moving_targets, restaurants
from src.restaurants.models.mt import MTLearner, MTMaster


class MT(moving_targets.MACS):
    def __init__(self,
                 monotonicities,
                 h_units=None,
                 scaler=None,
                 alpha=1.,
                 beta=1.,
                 restart_fit=True,
                 time_limit=30,
                 loss='mse',
                 optimizer='adam',
                 **kwargs):
        super(MT, self).__init__(
            learner=MTLearner(h_units, scaler, restart_fit, loss, optimizer, **kwargs),
            master=MTMaster(monotonicities, alpha, beta, time_limit),
            init_step='pretraining',
            metrics=[AUC(), GroundR2(self)]
        )


class AUC(moving_targets.Metric):
    def __init__(self):
        super(AUC, self).__init__('AUC')

    def __call__(self, x, y, pred):
        return roc_auc_score(y[y != -1], pred[y != -1])


class GroundR2(moving_targets.Metric):
    def __init__(self, macs: MT, res=100):
        super(GroundR2, self).__init__('Ground R2')
        ar, nr, dr = np.meshgrid(np.linspace(1, 5, num=res), np.linspace(0, 200, num=res), ['D', 'DD', 'DDD', 'DDDD'])
        self.avg_ratings = ar.reshape(-1, )
        self.num_reviews = nr.reshape(-1, )
        self.dollar_ratings = dr.reshape(-1, )
        self.ground_truth = restaurants.ctr_estimate(self.avg_ratings, self.num_reviews, self.dollar_ratings)
        self.macs = macs

    def __call__(self, x, y, pred):
        pred = self.macs.learner.model.ctr_estimate(self.avg_ratings, self.num_reviews, self.dollar_ratings)
        return r2_score(self.ground_truth, pred)
