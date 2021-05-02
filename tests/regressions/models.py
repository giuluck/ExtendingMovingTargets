from tensorflow.python.keras.callbacks import EarlyStopping

from src.models import MTLearner, MLP, MTMaster


class Learner(MTLearner):
    def __init__(self):
        def model():
            m = MLP(output_act=None, h_units=[16] * 4)
            m.compile(optimizer='adam', loss='mse')
            return m

        # similar to the default behaviour of the scikit MLP (tol = 1e-4, n_iter_no_change = 10, max_iter = 200)
        es = EarlyStopping(monitor='loss', patience=10, min_delta=1e-4)
        super(Learner, self).__init__(model, epochs=200, callbacks=[es], verbose=False)


class Master(MTMaster):
    def __init__(self, monotonicities, augmented_mask, learner_y='original', **kwargs):
        assert learner_y in ['original', 'augmented'], "learner_y should be either 'original' or 'augmented'"
        super(Master, self).__init__(monotonicities=monotonicities, augmented_mask=augmented_mask, **kwargs)
        self.learner_y = learner_y

    def return_solutions(self, macs, solution, model_info, x, y, iteration):
        adj, kwargs = super(Master, self).return_solutions(macs, solution, model_info, x, y, iteration)
        if self.learner_y == 'augmented':
            learner_y = adj.copy()
            learner_y[~self.augmented_mask] = y[~self.augmented_mask]
            kwargs['y'] = learner_y
        return adj, kwargs
