from abc import ABC

import numpy as np
from gurobipy import Model, Env, GRB

from moving_targets.masters.master import Master


class GurobiMaster(Master, ABC):
    def __init__(self, alpha=1., beta=1., time_limit=30.):
        super(GurobiMaster, self).__init__(alpha=alpha, beta=beta)
        self.time_limit = time_limit

    def adjust_targets(self, macs, x, y, iteration):
        # build model and get losses
        with Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('TimeLimit', self.time_limit)
            env.start()
            with Model(env=env, name='model') as model:
                model_info = self.build_model(macs, model, x, y, iteration)
                model.update()

                # algorithm core: check for feasibility and behave depending on that
                y_loss = self.y_loss(macs, model, model_info, x, y, iteration)
                model.update()
                p_loss = self.p_loss(macs, model, model_info, x, y, iteration)
                model.update()
                if self.beta_step(macs, model, model_info, x, y, iteration):
                    model.addConstr(p_loss <= self.beta, name='loss')
                    model.update()
                    model.setObjective(y_loss, GRB.MINIMIZE)
                    model.update()
                else:
                    model.setObjective(y_loss + (1.0 / self.alpha) * p_loss, GRB.MINIMIZE)
                    model.update()

                # solve the problem and get the adjusted labels
                model.optimize()
                if model.Status == GRB.INFEASIBLE:
                    raise RuntimeError('The given model has no admissible solution, please check its constraints.')
                return self.return_solutions(macs, model, model_info, x, y, iteration)

    @staticmethod
    def custom_loss(loss, model, numeric_variables, model_variables, sample_weight=None):
        # use uniform weights if none are passed, otherwise normalize the weights so that they sum to len(samples)
        if sample_weight is None:
            sample_weight = np.ones(len(numeric_variables))
        else:
            sample_weight = len(sample_weight) * np.array(sample_weight) / np.sum(sample_weight)
        return sum([sw * loss(nv, mv) for nv, mv, sw in zip(numeric_variables, model_variables, sample_weight)])

    @staticmethod
    def sum_of_absolute_errors(**kwargs):
        return GurobiMaster.custom_loss(
            loss=lambda nv, mv: abs(nv - mv),
            **kwargs
        )

    @staticmethod
    def sum_of_squared_errors(**kwargs):
        return GurobiMaster.custom_loss(
            loss=lambda nv, mv: (nv - mv) ** 2,
            **kwargs
        )

    @staticmethod
    def mean_absolute_error(numeric_variables, **kwargs):
        return GurobiMaster.sum_of_absolute_errors(
            numeric_variables=numeric_variables,
            **kwargs
        ) / len(numeric_variables)

    @staticmethod
    def mean_squared_error(numeric_variables, **kwargs):
        return GurobiMaster.sum_of_squared_errors(
            numeric_variables=numeric_variables,
            **kwargs
        ) / len(numeric_variables)

    @staticmethod
    def binary_hamming(numeric_variables, **kwargs):
        return GurobiMaster.custom_loss(
            loss=lambda nv, mv: nv * mv + (1 - nv) * mv,
            numeric_variables=numeric_variables,
            **kwargs
        ) / len(numeric_variables)

    @staticmethod
    def binary_crossentropy(numeric_variables, eps=1e-3, **kwargs):
        numeric_variables = np.clip(numeric_variables, a_min=eps, a_max=1 - eps)
        return GurobiMaster.custom_loss(
            loss=lambda nv, mv: -(mv * np.log(nv) + (1 - mv) * np.log(1 - nv)),
            numeric_variables=numeric_variables,
            **kwargs
        ) / len(numeric_variables)

    @staticmethod
    def swapped_binary_crossentropy(model, numeric_variables, **kwargs):
        return GurobiMaster.custom_loss(
            loss=lambda nv, mv: -(nv * model.log(mv) + (1 - nv) * model.log(mv)),
            model=model,
            numeric_variables=numeric_variables,
            **kwargs
        )

    @staticmethod
    def categorical_hamming(numeric_variables, **kwargs):
        return GurobiMaster.custom_loss(
            loss=lambda nv, mv: 1 - mv[nv],
            numeric_variables=numeric_variables,
            **kwargs
        ) / len(numeric_variables)

    @staticmethod
    def categorical_crossentropy(numeric_variables, eps=1e-3, **kwargs):
        numeric_variables = np.clip(numeric_variables, a_min=eps, a_max=1 - eps)
        return GurobiMaster.custom_loss(
            loss=lambda nv, mv: -sum(mv * np.log(nv)),
            numeric_variables=numeric_variables,
            **kwargs
        ) / len(numeric_variables)
