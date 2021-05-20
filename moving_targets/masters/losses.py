from collections import Callable

import numpy as np


class SumLoss:
    def __init__(self, call_fn: Callable, loss_fn: Callable):
        self.call_fn: Callable = call_fn
        self.loss_fn: Callable = loss_fn

    def __call__(self, model, numeric_variables, model_variables, sample_weight=None):
        return self.call_fn(
            loss=self.loss_fn,
            model=model,
            numeric_variables=numeric_variables,
            model_variables=model_variables,
            sample_weight=sample_weight
        )


class MeanLoss(SumLoss):
    def __call__(self, model, numeric_variables, model_variables, sample_weight=None):
        return super(MeanLoss, self).__call__(
            model=model,
            numeric_variables=numeric_variables,
            model_variables=model_variables,
            sample_weight=sample_weight
        ) / len(numeric_variables)


class ClippedMeanLoss(MeanLoss):
    def __init__(self, call_fn: Callable, loss_fn: Callable, clip_value: float = 1e-15):
        super(ClippedMeanLoss, self).__init__(call_fn=call_fn, loss_fn=loss_fn)
        self.clip_value: float = clip_value

    def __call__(self, model, numeric_variables, model_variables, sample_weight=None):
        numeric_variables = np.clip(numeric_variables, a_min=self.clip_value, a_max=1 - self.clip_value)
        return super(ClippedMeanLoss, self).__call__(
            model=model,
            numeric_variables=numeric_variables,
            model_variables=model_variables,
            sample_weight=sample_weight
        )


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class LossesHandler:
    def __init__(self,
                 sum_fn=lambda model, x: model.sum(x),
                 abs_fn=lambda model, x: model.abs(x),
                 log_fn=lambda model, x: model.log(x)):
        # metadata
        self.sum_fn = sum_fn
        self.abs_fn = abs_fn
        self.log_fn = log_fn
        # losses
        self.sum_of_absolute_errors = SumLoss(call_fn=self.__call__, loss_fn=self.absolute_errors)
        self.sum_of_squared_errors = SumLoss(call_fn=self.__call__, loss_fn=self.squared_errors)
        self.mean_absolute_error = MeanLoss(call_fn=self.__call__, loss_fn=self.absolute_errors)
        self.mean_squared_error = MeanLoss(call_fn=self.__call__, loss_fn=self.squared_errors)
        self.binary_hamming = MeanLoss(call_fn=self.__call__, loss_fn=self.binary_hamming)
        self.binary_crossentropy = ClippedMeanLoss(call_fn=self.__call__, loss_fn=self.binary_crossentropy)
        self.swapped_binary_crossentropy = MeanLoss(call_fn=self.__call__, loss_fn=self.swapped_binary_crossentropy)
        self.categorical_hamming = MeanLoss(call_fn=self.__call__, loss_fn=self.categorical_hamming)
        self.categorical_crossentropy = ClippedMeanLoss(call_fn=self.__call__, loss_fn=self.categorical_crossentropy)

    def absolute_errors(self, model, numeric_variable, model_variable):
        return self.abs_fn(model, numeric_variable - model_variable)

    def squared_errors(self, model, numeric_variable, model_variable):
        return (numeric_variable - model_variable) ** 2

    def binary_hamming(self, model, numeric_variable, model_variable):
        return numeric_variable * model_variable + (1 - numeric_variable) * model_variable

    def binary_crossentropy(self, model, numeric_variable, model_variable):
        return -(model_variable * np.log(numeric_variable) + (1 - model_variable) * np.log(1 - numeric_variable))

    def swapped_binary_crossentropy(self, model, numeric_variable, model_variable):
        _log = lambda x: self.log_fn(model, x)
        return -(numeric_variable * _log(model_variable) + (1 - numeric_variable) * _log(1 - model_variable))

    def categorical_hamming(self, model, numeric_variable, model_variable):
        return 1 - model_variable[numeric_variable]

    def categorical_crossentropy(self, model, numeric_variable, model_variable):
        return self.sum_fn(model, -model_variable * np.log(numeric_variable))

    def __call__(self, loss, model, numeric_variables, model_variables, sample_weight):
        # use uniform weights if none are passed, otherwise normalize the weights so that they sum to len(samples)
        if sample_weight is None:
            sample_weight = np.ones(len(numeric_variables))
        else:
            sample_weight = len(sample_weight) * np.array(sample_weight) / np.sum(sample_weight)
        # compute sum of losses
        variables = zip(numeric_variables, model_variables, sample_weight)
        return self.sum_fn(model, [sw * loss(model, nv, mv) for nv, mv, sw in variables])
