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
    weight_methods = ['uniform', 'distance', 'gamma', 'feasibility-prop', 'feasibility-step']
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
        if self.weight_method == 'uniform':
            # no sample weights
            return adj_y
        if self.weight_method == 'gamma':
            # each ground sample is weighted 1.0, while each augmented sample is weighted 1.0 / gamma
            # (if gamma is the number of augmented samples, the totality of them is weighted as the original one)
            return adj_y, {'sample_weight': np.ones_like(y) / self.gamma}
        if self.weight_method in ['distance', 'feasibility-prop', 'feasibility-step']:
            _, pred = model_info
            mask = np.isnan(y)
            # COMPUTATION
            sample_weight = np.zeros_like(pred)
            if self.weight_method == 'distance':
                # sample weights are directly proportional to the distance from the adjusted target to the prediction
                # (if adj_y == p then that sample is pretty useless)
                sample_weight = np.abs(adj_y - pred)
            else:
                vls = np.maximum(0.0, pred[self.lower_indices] - pred[self.higher_indices])
                vls = vls if self.weight_method == 'feasibility-prop' else np.where(vls == 0.0, 0.0, 1.0 / self.gamma)
                for vl, hi, li in zip(vls, self.higher_indices, self.lower_indices):
                    sample_weight[hi] += vl
                    sample_weight[li] += vl
            # NORMALIZATION
            if sample_weight[mask].max() == 0.0:
                # if the maximum is zero, adopt gamma policy
                sample_weight = np.ones_like(y) / self.gamma
            else:
                # if the maximum is non-zero, normalize into [min_weight, 1]
                sample_weight = sample_weight / sample_weight[mask].max()
                sample_weight = (1 - self.min_weight) * sample_weight + self.min_weight
            # give total weight to original samples
            sample_weight[~mask] = 1.0
            return adj_y, {'sample_weight': sample_weight}
        return NotImplementedError(f'sample_weight {self.weight_method} can not be handled')
