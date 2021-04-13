import numpy as np
from sklearn.neural_network import MLPRegressor
from tensorflow.python.keras.callbacks import EarlyStopping

from src.models import MTLearner, MLP, MTMaster, MT


# ------------------------------------------------------ LEARNERS ------------------------------------------------------
class TestMTL(MTLearner):
    pass


class Keras(TestMTL):
    def __init__(self, optimizer='adam', warm_start=False, verbose=False):
        def model():
            m = MLP(output_act=None, h_units=[16] * 4)
            m.compile(optimizer=optimizer, loss='mse')
            return m

        # similar to the default behaviour of the scikit MLP (tol = 1e-4, n_iter_no_change = 10, max_iter = 200)
        es = EarlyStopping(monitor='loss', patience=10, min_delta=1e-4)
        super(Keras, self).__init__(model, warm_start=warm_start, epochs=200, verbose=verbose, callbacks=[es])


class Scikit(TestMTL):
    def __init__(self, solver='adam', warm_start=False, verbose=False):
        def model():
            return MLPRegressor([16] * 4, solver=solver, warm_start=warm_start, verbose=verbose)

        super(Scikit, self).__init__(model, warm_start=True)


# ------------------------------------------------------ MASTERS ------------------------------------------------------
class TestMTM(MTMaster):
    def __init__(self, monotonicities, prop_beta=True, loss_fn='mae', alpha=1., beta=1.):
        super(TestMTM, self).__init__(monotonicities=monotonicities, loss_fn=loss_fn, alpha=alpha, beta=beta)
        self.base_beta = beta if prop_beta else None

    def y_loss(self, macs, model, model_info, x, y, iteration):
        y_loss = super(TestMTM, self).y_loss(macs, model, model_info, x, y, iteration)
        if self.base_beta is not None:
            self.beta = self.base_beta * y_loss
        return y_loss

    def is_feasible(self, macs, model, model_info, x, y, iteration):
        is_feasible = super(TestMTM, self).is_feasible(macs, model, model_info, x, y, iteration)
        return is_feasible


class Uniform(TestMTM):
    pass


class DistanceProportional(TestMTM):
    def return_solutions(self, macs, solution, model_info, x, y, iteration):
        adj_y = super(TestMTM, self).return_solutions(macs, solution, model_info, x, y, iteration)
        # sample weights are directly proportional to the distance from the adjusted target to the prediction
        # (if adj_y == p then that sample is pretty useless)
        _, pred = model_info
        mask = np.isnan(y)
        sample_weight = np.abs(adj_y - pred)
        if sample_weight[mask].max() > 0:
            sample_weight = sample_weight / sample_weight[mask].max()
        sample_weight[~mask] = 1.0
        return adj_y, {'sample_weight': sample_weight}


# -------------------------------------------------------- MACS --------------------------------------------------------
class TestMT(MT):
    def on_iteration_start(self, macs, x, y, val_data, iteration):
        print(f'-------------------- ITERATION: {iteration:02} --------------------')
        super(TestMT, self).on_iteration_start(macs, x, y, val_data, iteration)

    def on_iteration_end(self, macs, x, y, val_data, iteration):
        super(TestMT, self).on_iteration_end(macs, x, y, val_data, iteration)
        print(f'Time: {self.cache["time/iteration"]:.4f} s')
