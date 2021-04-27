import numpy as np
from sklearn.neural_network import MLPRegressor
from tensorflow.python.keras.callbacks import EarlyStopping

from src.models import MTLearner, MLP, MTMaster


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


class Master(MTMaster):
    weight_method = ['uniform', 'distance', 'omega', 'feasibility-prop', 'feasibility-step', 'feasibility-omega',
                     'memory-prop', 'memory-step', 'memory-omega', 'memory-inc']
    beta_methods = ['none', 'standard', 'proportional']
    pert_methods = ['none', 'loss', 'constraint']

    def __init__(self, monotonicities, augmented_mask, weight_method='uniform', omega_learner=1.0, omega_master=None,
                 min_weight=None, beta_method='standard', perturbation_method='none', perturbation=None, **kwargs):
        assert weight_method in self.weight_method, f'sample_weight should be in {self.weight_method}'
        assert beta_method in self.beta_methods, f'beta_method should be in {self.beta_methods}'
        assert perturbation_method in self.pert_methods, f'perturbation_method should be in {self.pert_methods}'
        super(Master, self).__init__(monotonicities=monotonicities, **kwargs)
        self.augmented_mask = augmented_mask
        self.weight_method = weight_method
        self.omega_learner = omega_learner
        self.omega_master = omega_learner if omega_master is None else omega_master
        self.min_weight = 1.0 / omega_learner if min_weight is None else min_weight
        self.base_beta = self.beta
        self.beta_method = beta_method
        self.perturbation_method = perturbation_method
        self.perturbation = perturbation
        self.weights_memory = np.zeros(len(augmented_mask))

    def build_model(self, macs, model, x, y, iteration):
        var, pred = super(Master, self).build_model(macs, model, x, y, iteration)
        if self.perturbation_method == 'constraint':
            # for each variable/prediction pair, we force them to be at least randomly distant from the model prediction
            absolute_perturbation = (y.max() - y.min()) * self.perturbation
            distances = np.abs(np.random.normal(scale=absolute_perturbation, size=len(pred[self.augmented_mask])))
            for v, p, d in zip(var[self.augmented_mask], pred[self.augmented_mask], distances):
                model.add(model.abs(v - p) >= d)
        return var, pred

    def y_loss(self, macs, model, model_info, x, y, iteration):
        var, _ = model_info
        y_loss = self.loss_fn(model, y[~np.isnan(y)], var[~np.isnan(y)])
        if self.beta_method == 'proportional':
            self.beta = self.base_beta * y_loss
        return y_loss

    def p_loss(self, macs, model, model_info, x, y, iteration):
        sw = np.where(self.augmented_mask, self.omega_master, 1)
        var, pred = model_info
        if self.perturbation_method == 'loss':
            # before computing the loss, we perturb the predictions
            absolute_perturbation = (y.max() - y.min()) * self.perturbation
            pred = pred + np.random.normal(scale=absolute_perturbation, size=len(pred))
        return 0.0 if pred is None else self.loss_fn(model, pred, var, sample_weight=sw)

    def is_feasible(self, macs, model, model_info, x, y, iteration):
        feasible = super(Master, self).is_feasible(macs, model, model_info, x, y, iteration)
        feasible = False if self.beta_method is 'none' else feasible
        macs.log(**{'master/method': int(feasible)})
        return feasible

    def return_solutions(self, macs, solution, model_info, x, y, iteration):
        adj_y = super(Master, self).return_solutions(macs, solution, model_info, x, y, iteration)
        _, pred = model_info
        if self.weight_method == 'uniform':
            # no sample weights
            return adj_y
        elif self.weight_method == 'omega':
            # each ground sample is weighted 1.0, while each augmented sample is weighted 1.0 / omega
            # (if omega is the number of augmented samples, the totality of them is weighted as the original one)
            sample_weight = np.where(self.augmented_mask, 1.0 / self.omega_learner, 1.0)
        elif self.weight_method == 'distance':
            # sample weights are directly proportional to the distance from the adjusted target to the prediction
            # (if adj_y == p then that sample is pretty useless)
            sample_weight = np.abs(adj_y - pred)
            sample_weight = self.normalize(sample_weight)
        elif 'feasibility' in self.weight_method or 'memory' in self.weight_method:
            if 'memory' not in self.weight_method:
                self.weights_memory = np.zeros_like(pred)
            diffs = pred[self.lower_indices] - pred[self.higher_indices]
            if 'omega' in self.weight_method:
                # in case of feasibility- or memory-omega, the augmented points get a value of 1 / omega independently
                # from the number and the value of the violations
                for df, hi, li in zip(diffs[diffs > 0], self.higher_indices[diffs > 0], self.lower_indices[diffs > 0]):
                    self.weights_memory[hi] = 1.0 / self.omega_learner
                    self.weights_memory[li] = 1.0 / self.omega_learner
                sample_weight = self.weights_memory.copy()
                # if there is no violation, adopt omega policy
                if sample_weight[self.augmented_mask].max() == 0.0:
                    sample_weight = np.ones_like(y) / self.omega_learner
            elif 'inc' in self.weight_method:
                # in case of memory-inc, the augmented points get a value of 1 / omega independently at each iteration
                for df, hi, li in zip(diffs[diffs > 0], self.higher_indices[diffs > 0], self.lower_indices[diffs > 0]):
                    self.weights_memory[hi] += 1.0 / self.omega_learner
                    self.weights_memory[li] += 1.0 / self.omega_learner
                sample_weight = self.weights_memory.copy()
                # if there is no violation, adopt omega policy
                if sample_weight[self.augmented_mask].max() == 0.0:
                    sample_weight = np.ones_like(y) / self.omega_learner
            elif 'step' in self.weight_method or 'prop' in self.weight_method:
                # differently from feasibility- and memory-prop, in case of feasibility- or memory-step
                # the increase is constant (1 / omega)
                if 'step' in self.weight_method:
                    diffs = np.where(diffs > 0, 1 / self.omega_learner, 0.0)
                for df, hi, li in zip(diffs[diffs > 0], self.higher_indices[diffs > 0], self.lower_indices[diffs > 0]):
                    self.weights_memory[hi] += df
                    self.weights_memory[li] += df
                sample_weight = self.normalize(self.weights_memory.copy())
            else:
                raise NotImplementedError(f'sample_weight {self.weight_method} can not be handled')
        else:
            raise NotImplementedError(f'sample_weight {self.weight_method} can not be handled')
        sample_weight[~self.augmented_mask] = 1.0
        return adj_y, {'sample_weight': sample_weight}

    def normalize(self, sw):
        if sw[self.augmented_mask].max() == 0.0:
            # if the maximum is zero, adopt omega policy
            sw = np.ones_like(sw) / self.omega_learner
        else:
            # if the maximum is non-zero, normalize into [min_weight, 1]
            sw = sw / sw[self.augmented_mask].max()
            sw = (1 - self.min_weight) * sw + self.min_weight
        return sw
