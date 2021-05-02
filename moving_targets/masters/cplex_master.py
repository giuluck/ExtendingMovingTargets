import numpy as np
from docplex.mp.model import Model as CPModel

from moving_targets.masters.master import Master


class CplexMaster(Master):
    def __init__(self, alpha=1., beta=1., time_limit=30.):
        super(CplexMaster, self).__init__(alpha=alpha, beta=beta)
        self.time_limit = time_limit

    def build_model(self, macs, model, x, y, iteration):
        raise NotImplementedError("Please implement method 'build_model'")

    def beta_step(self, macs, model, model_info, x, y, iteration):
        return False

    def y_loss(self, macs, model, model_info, x, y, iteration):
        return 0.0

    def p_loss(self, macs, model, model_info, x, y, iteration):
        return 0.0

    def return_solutions(self, macs, solution, model_info, x, y, iteration):
        return NotImplementedError("Please implement method 'return_solutions'")

    def adjust_targets(self, macs, x, y, iteration):
        # build model and get losses
        model = CPModel()
        model.set_time_limit(self.time_limit)
        model_info = self.build_model(macs, model, x, y, iteration)

        # algorithm core: check for feasibility and behave depending on that
        beta_step = self.beta_step(macs, model, model_info, x, y, iteration)
        y_loss = self.y_loss(macs, model, model_info, x, y, iteration)
        p_loss = self.p_loss(macs, model, model_info, x, y, iteration)
        if beta_step:
            model.add(p_loss <= self.beta)
            model.minimize(y_loss)
        else:
            model.minimize(y_loss + (1.0 / self.alpha) * p_loss)

        # solve the problem and get the adjusted labels
        solution = model.solve()
        if solution is None:
            raise RuntimeError('The given model has no admissible solution, please check the implemented constraints.')
        return self.return_solutions(macs, solution, model_info, x, y, iteration)

    @staticmethod
    def custom_loss(model, loss, numeric_variables, model_variables, sample_weight=None):
        # use uniform weights if none are passed, otherwise normalize the weights so that they sum to len(samples)
        if sample_weight is None:
            sample_weight = np.ones(len(numeric_variables))
        else:
            sample_weight = len(sample_weight) * np.array(sample_weight) / np.sum(sample_weight)
        return model.sum([sw * loss(nv, mv) for nv, mv, sw in zip(numeric_variables, model_variables, sample_weight)])

    @staticmethod
    def sum_of_absolute_errors(model, numeric_variables, model_variables, sample_weight=None):
        return CplexMaster.custom_loss(model=model,
                                       loss=lambda nv, mv: model.abs(nv - mv),
                                       numeric_variables=numeric_variables,
                                       model_variables=model_variables,
                                       sample_weight=sample_weight)

    @staticmethod
    def sum_of_squared_errors(model, numeric_variables, model_variables, sample_weight=None):
        return CplexMaster.custom_loss(model=model,
                                       loss=lambda nv, mv: (nv - mv) ** 2,
                                       numeric_variables=numeric_variables,
                                       model_variables=model_variables,
                                       sample_weight=sample_weight)

    @staticmethod
    def mean_absolute_error(model, numeric_variables, model_variables, sample_weight=None):
        return CplexMaster.sum_of_absolute_errors(model=model,
                                                  numeric_variables=numeric_variables,
                                                  model_variables=model_variables,
                                                  sample_weight=sample_weight) / len(numeric_variables)

    @staticmethod
    def mean_squared_error(model, numeric_variables, model_variables, sample_weight=None):
        return CplexMaster.sum_of_squared_errors(model=model,
                                                 numeric_variables=numeric_variables,
                                                 model_variables=model_variables,
                                                 sample_weight=sample_weight) / len(numeric_variables)

    @staticmethod
    def binary_crossentropy(model, numeric_variables, model_variables, sample_weight=None, eps=1e-3):
        numeric_variables = np.clip(numeric_variables, a_min=eps, a_max=1 - eps)
        return CplexMaster.custom_loss(model=model,
                                       loss=lambda nv, mv: -(mv * np.log(nv) + (1 - mv) * np.log(1 - nv)),
                                       numeric_variables=numeric_variables,
                                       model_variables=model_variables,
                                       sample_weight=sample_weight) / len(numeric_variables)

    @staticmethod
    def binary_indicator(model, numeric_variables, model_variables, sample_weight=None):
        return CplexMaster.custom_loss(model=model,
                                       loss=lambda nv, mv: nv * mv + (1 - nv) * mv,
                                       numeric_variables=numeric_variables,
                                       model_variables=model_variables,
                                       sample_weight=sample_weight) / len(numeric_variables)

    @staticmethod
    def categorical_crossentropy(model, numeric_variables, model_variables, sample_weight=None, eps=1e-3):
        numeric_variables = np.clip(numeric_variables, a_min=eps, a_max=1 - eps)
        return CplexMaster.custom_loss(model=model,
                                       loss=lambda nv, mv: -model.sum(mv * np.log(nv)),
                                       numeric_variables=numeric_variables,
                                       model_variables=model_variables,
                                       sample_weight=sample_weight) / len(numeric_variables)

    @staticmethod
    def categorical_indicator(model, numeric_variables, model_variables, sample_weight=None):
        return CplexMaster.custom_loss(model=model,
                                       loss=lambda nv, mv: 1 - mv[nv],
                                       numeric_variables=numeric_variables,
                                       model_variables=model_variables,
                                       sample_weight=sample_weight) / len(numeric_variables)
