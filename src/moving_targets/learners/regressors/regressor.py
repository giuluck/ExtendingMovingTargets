from src.moving_targets.learners import Learner


class Regressor(Learner):
    def __init__(self):
        super(Regressor, self).__init__()

    def predict_proba(self, x):
        raise NotImplementedError('Regressors do not support probabilities')
