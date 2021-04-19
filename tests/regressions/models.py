import numpy as np
from sklearn.neural_network import MLPRegressor
from tensorflow.python.keras.callbacks import EarlyStopping

from src.models import MTLearner, MLP, MTMaster
from src.util.augmentation import filter_vectors


class Learner(MTLearner):
    def __init__(self, backend='keras', optimizer='adam', warm_start=False, verbose=False):
        if backend == 'keras':
            def model():
                m = MLP(output_act=None, h_units=[16] * 4)
                m.compile(optimizer=optimizer, loss='mse')
                return m

            # similar to the default behaviour of the scikit MLP (tol = 1e-4, n_iter_no_change = 10, max_iter = 200)
            es = EarlyStopping(monitor='loss', patience=10, min_delta=1e-4)
            super(Learner, self).__init__(model, warm_start=warm_start, epochs=200, verbose=verbose, callbacks=[es])
        elif backend == 'scikit':
            def model():
                return MLPRegressor([16] * 4, solver=optimizer, warm_start=warm_start, verbose=verbose)

            super(Learner, self).__init__(model, warm_start=True)
        else:
            raise ValueError("backend should be either 'keras' or 'scikit'")


class UnsupervisedMaster(MTMaster):
    weight_methods = ['uniform', 'distance', 'gamma', 'feasibility-prop', 'feasibility-step', 'feasibility-same']
    beta_methods = ['none', 'standard', 'proportional']
    pert_methods = ['none', 'loss', 'constraint']

    def __init__(self, monotonicities, alpha=1.0, beta=1.0, loss_fn='mae', weight_method='uniform', gamma=None,
                 min_weight='gamma', beta_method='standard', perturbation_method='none', perturbation=None):
        assert weight_method in self.weight_methods, f'sample_weight should be in {self.weight_methods}'
        assert beta_method in self.beta_methods, f'beta_method should be in {self.beta_methods}'
        assert perturbation_method in self.pert_methods, f'perturbation_method should be in {self.pert_methods}'
        super(UnsupervisedMaster, self).__init__(monotonicities=monotonicities, alpha=alpha, beta=beta, loss_fn=loss_fn)
        self.weight_method = weight_method
        self.gamma = gamma
        self.min_weight = (0.0 if gamma is None else 1.0 / gamma) if min_weight == 'gamma' else min_weight
        self.base_beta = self.beta
        self.beta_method = beta_method
        self.perturbation_method = perturbation_method
        self.perturbation = perturbation

    def build_model(self, macs, model, x, y, iteration):
        var, pred = super(UnsupervisedMaster, self).build_model(macs, model, x, y, iteration)
        if self.perturbation_method == 'constraint':
            # for each variable/prediction pair, we force them to be at least randomly distant from the model prediction
            absolute_perturbation = (y.max() - y.min()) * self.perturbation
            _, vv, pp = filter_vectors(self.mask_value, y, var, pred)
            dd = np.abs(np.random.normal(scale=absolute_perturbation, size=len(pp)))
            for v, p, d in zip(vv, pp, dd):
                model.add(model.abs(v - p) >= d)
        return var, pred

    def y_loss(self, macs, model, model_info, x, y, iteration):
        y_loss = super(UnsupervisedMaster, self).y_loss(macs, model, model_info, x, y, iteration)
        if self.beta_method is 'proportional':
            self.beta = self.base_beta * y_loss
        return y_loss

    def p_loss(self, macs, model, model_info, x, y, iteration):
        var, pred = model_info
        if self.perturbation_method == 'loss':
            # before computing the loss, we perturb the predictions
            absolute_perturbation = (y.max() - y.min()) * self.perturbation
            pred = pred + np.random.normal(scale=absolute_perturbation, size=len(pred))
        return 0.0 if pred is None else self.loss_fn(model, pred, var)

    def is_feasible(self, macs, model, model_info, x, y, iteration):
        feasible = super(UnsupervisedMaster, self).is_feasible(macs, model, model_info, x, y, iteration)
        feasible = False if self.beta_method is 'none' else feasible
        macs.log(**{'master/method': int(feasible)})
        return feasible

    def return_solutions(self, macs, solution, model_info, x, y, iteration):
        adj_y = super(UnsupervisedMaster, self).return_solutions(macs, solution, model_info, x, y, iteration)
        _, pred = model_info
        if self.weight_method == 'uniform':
            # no sample weights
            return adj_y
        elif self.weight_method == 'gamma':
            # each ground sample is weighted 1.0, while each augmented sample is weighted 1.0 / gamma
            # (if gamma is the number of augmented samples, the totality of them is weighted as the original one)
            sample_weight = np.ones_like(y) / self.gamma
        elif self.weight_method == 'distance':
            # sample weights are directly proportional to the distance from the adjusted target to the prediction
            # (if adj_y == p then that sample is pretty useless)
            sample_weight = np.abs(adj_y - pred)
            sample_weight = self.normalize(sample_weight, np.isnan(y))
        elif 'feasibility' in self.weight_method:
            sample_weight = np.zeros_like(pred)
            diffs = pred[self.lower_indices] - pred[self.higher_indices]
            if self.weight_method == 'feasibility-same':
                # in case of feasibility-same, the augmented points get a value of 1 / gamma independently from the
                # number and the value of the violations
                for df, hi, li in zip(diffs[diffs > 0], self.higher_indices[diffs > 0], self.lower_indices[diffs > 0]):
                    sample_weight[hi] = 1.0 / self.gamma
                    sample_weight[li] = 1.0 / self.gamma
                # in case of feasibility, assign 1 / gamma to each sample
                if sample_weight[np.isnan(y)].max() == 0.0:
                    sample_weight = np.ones_like(np.isnan(y)) / self.gamma
                sample_weight[~np.isnan(y)] = 1.0
            else:
                # differently from feasibility-prop, in case of feasibility-step the increase is constant (1 / gamma)
                if self.weight_method == 'feasibility-step':
                    diffs = np.where(diffs > 0, 1 / self.gamma, 0.0)
                for df, hi, li in zip(diffs[diffs > 0], self.higher_indices[diffs > 0], self.lower_indices[diffs > 0]):
                    sample_weight[hi] += df
                    sample_weight[li] += df
                sample_weight = self.normalize(sample_weight, np.isnan(y))
        else:
            raise NotImplementedError(f'sample_weight {self.weight_method} can not be handled')
        return adj_y, {'sample_weight': sample_weight}

    def normalize(self, sw, mask):
        if sw[mask].max() == 0.0:
            # if the maximum is zero, adopt gamma policy
            sw = np.ones_like(sw) / self.gamma
        else:
            # if the maximum is non-zero, normalize into [min_weight, 1]
            sw = sw / sw[mask].max()
            sw = (1 - self.min_weight) * sw + self.min_weight
        sw[~mask] = 1.0
        return sw
