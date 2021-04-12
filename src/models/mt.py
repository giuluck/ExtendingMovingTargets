import time
import numpy as np

from moving_targets import MACS
from moving_targets.learners import Learner
from moving_targets.masters import CplexMaster
from src.models.model import Model
from src.util.augmentation import filter_vectors


class MTLearner(Learner):
    def __init__(self, build_model, warm_start=False, mask_value=np.nan, **kwargs):
        super(MTLearner, self).__init__()
        self.build_model = build_model
        self.warm_start = warm_start
        self.mask_value = mask_value
        self.fit_args = kwargs
        self.model = build_model()

    def fit(self, macs, x, y, iteration, **kwargs):
        start_time = time.time()
        y, x = filter_vectors(self.mask_value, y, x)
        if not self.warm_start:
            self.model = self.build_model()
        fit = self.model.fit(x, y, **self.fit_args, **kwargs)
        # retrieve number of epochs and last loss depending on the model
        if 'keras' in str(type(fit)):
            epochs, loss = fit.epoch[-1] + 1, fit.history['loss'][-1]
        elif 'sklearn' in str(type(fit)):
            epochs, loss = fit.n_iter_, fit.loss_
        else:
            epochs, loss = np.nan, np.nan
        # log info
        macs.log(**{
            'time/learner': time.time() - start_time,
            'learner/epochs': epochs,
            'learner/loss': loss
        })

    def predict(self, x):
        return self.model.predict(x).reshape(-1, )


class MTMaster(CplexMaster):
    def __init__(self, monotonicities, loss_fn='mae', alpha=1., beta=1., mask_value=np.nan, time_limit=30):
        super(MTMaster, self).__init__(alpha=alpha, beta=beta, time_limit=time_limit)
        assert loss_fn in ['mae', 'mse', 'sae', 'sse'], "Loss should be one in ['mae', 'mse', 'sae', 'sse']"
        self.loss_fn = getattr(CplexMaster, f'{loss_fn}_loss')
        self.higher_indices = np.array([hi for hi, _ in monotonicities])
        self.lower_indices = np.array([li for _, li in monotonicities])
        self.mask_value = mask_value

    def build_model(self, macs, model, x, y, iteration):
        # handle 'projection' initial step (p = None)
        pred = None if not macs.fitted else macs.predict(x)
        # create variables and impose constraints for each monotonicity
        var = np.array(model.continuous_var_list(keys=len(y), name='y'))
        if len(self.higher_indices):
            model.add_constraints([h >= l for h, l in zip(var[self.higher_indices], var[self.lower_indices])])
        # return model info
        return var, pred

    def is_feasible(self, macs, model, model_info, x, y, iteration):
        _, pred = model_info
        if len(self.higher_indices) == 0 or pred is None:
            violations = np.array([0])
        else:
            violations = np.maximum(0.0, pred[self.lower_indices] - pred[self.higher_indices])
        satisfied = violations == 0
        macs.log(**{
            'master/avg. violation': np.mean(violations),
            'master/pct. violation': 1 - np.mean(satisfied),
            'master/is feasible': int(satisfied.all())
        })
        return satisfied.all()

    def y_loss(self, macs, model, model_info, x, y, iteration):
        var, _ = model_info
        y_masked, var_masked = filter_vectors(self.mask_value, y, var)
        return self.loss_fn(model, y_masked, var_masked)

    def p_loss(self, macs, model, model_info, x, y, iteration):
        var, pred = model_info
        return 0.0 if pred is None else self.loss_fn(model, pred, var)

    def return_solutions(self, macs, solution, model_info, x, y, iteration):
        var, _ = model_info
        adj_y = np.array([vy.solution_value for vy in var])
        ground, adj_ground = filter_vectors(self.mask_value, y, adj_y)
        macs.log(**{
            'master/adj. mae': np.abs(adj_ground - ground).mean(),
            'master/adj. mse': np.mean((adj_ground - ground) ** 2),
            'time/master': solution.solve_details.time
        })
        # TODO: compute sample weights and return adjusted labels and sample weights
        return adj_y


class MT(MACS, Model):
    def __init__(self, learner, master, init_step='pretraining', metrics=None):
        super(MT, self).__init__(learner=learner, master=master, init_step=init_step, metrics=metrics)

    def on_iteration_end(self, macs, x, y, val_data, iteration):
        iteration = 0 if iteration == 'pretraining' else iteration
        logs = {'iteration': iteration, 'time/iteration': time.time() - self.time}
        for name, (xx, yy) in val_data.items():
            pp = self.predict(xx)
            for metric in self.metrics:
                logs[f'metrics/{name}_{metric.__name__}'] = metric(xx, yy, pp)
        self.log(**logs)
