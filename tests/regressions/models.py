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
    def __init__(self, monotonicities, gamma=None, prop_beta=True, loss_fn='mae', alpha=1., beta=1.):
        super(TestMTM, self).__init__(monotonicities=monotonicities, loss_fn=loss_fn, alpha=alpha, beta=beta)
        self.gamma = gamma
        self.base_beta = beta if prop_beta else None

    def y_loss(self, macs, model, model_info, x, y, iteration):
        y_loss = super(TestMTM, self).y_loss(macs, model, model_info, x, y, iteration)
        if self.base_beta is not None:
            self.beta = self.base_beta * y_loss
        return y_loss


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


class GammaWeighted(TestMTM):
    def return_solutions(self, macs, solution, model_info, x, y, iteration):
        adj_y = super(TestMTM, self).return_solutions(macs, solution, model_info, x, y, iteration)
        # each ground sample is weighted 1.0, while each augmented sample is weighted 1.0 / gamma
        # (if gamma is the number of augmented samples, the totality of them is weighted exactly as the original one)
        sample_weight = np.where(np.isnan(y), 1.0 / self.gamma, 1.0)
        return adj_y, {'sample_weight': sample_weight}


class FeasibilityProportional(TestMTM):
    # noinspection DuplicatedCode
    def return_solutions(self, macs, solution, model_info, x, y, iteration):
        adj_y = super(TestMTM, self).return_solutions(macs, solution, model_info, x, y, iteration)
        _, pred = model_info
        mask = np.isnan(y)
        sample_weight = np.zeros_like(y)
        violations = np.maximum(0.0, pred[self.lower_indices] - pred[self.higher_indices])
        for vl, hi, li in zip(violations, self.higher_indices, self.lower_indices):
            sample_weight[hi] += vl
            sample_weight[li] += vl
        # IN CASE OF FEASIBILITY ADOPT GAMMA-WEIGHTED POLICY (OTHERWISE NORMALIZE)
        if np.all(sample_weight[mask] == 0.0):
            sample_weight = np.ones_like(y) / self.gamma
        else:
            sample_weight = sample_weight / sample_weight[mask].max()
        sample_weight[~mask] = 1.0
        return adj_y, {'sample_weight': sample_weight}


class FeasibilityGamma(TestMTM):
    # noinspection DuplicatedCode
    def return_solutions(self, macs, solution, model_info, x, y, iteration):
        adj_y = super(TestMTM, self).return_solutions(macs, solution, model_info, x, y, iteration)
        _, pred = model_info
        mask = np.isnan(y)
        sample_weight = np.zeros_like(y)
        differences = pred[self.lower_indices] - pred[self.higher_indices]
        for diff, hi, li in zip(differences, self.higher_indices, self.lower_indices):
            if diff > 0:
                sample_weight[hi] += 1.0 / self.gamma
                sample_weight[li] += 1.0 / self.gamma
        # IN CASE OF FEASIBILITY ADOPT GAMMA-WEIGHTED POLICY
        if np.all(sample_weight[mask] == 0.0):
            sample_weight = np.ones_like(y) / self.gamma
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
