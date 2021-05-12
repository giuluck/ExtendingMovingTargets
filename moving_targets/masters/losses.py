import numpy as np


class SumLoss:
    def __init__(self, call_fn, loss_fn):
        self.call_fn = call_fn
        self.loss_fn = loss_fn

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
    def __init__(self, call_fn, loss_fn, eps=1e-3):
        super(ClippedMeanLoss, self).__init__(call_fn=call_fn, loss_fn=loss_fn)
        self.eps = eps

    def __call__(self, model, numeric_variables, model_variables, sample_weight=None):
        numeric_variables = np.clip(numeric_variables, a_min=self.eps, a_max=1 - self.eps)
        return super(MeanLoss, self).__call__(
            model=model,
            numeric_variables=numeric_variables,
            model_variables=model_variables,
            sample_weight=sample_weight
        ) / len(numeric_variables)


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class LossesHandler:
    def __init__(self, sum_fn=lambda model: sum, abs_fn=lambda model: abs, log_fn=lambda model: model.log):
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
        _abs = self.abs_fn(model)
        return _abs(numeric_variable - model_variable)

    def squared_errors(self, model, numeric_variable, model_variable):
        return (numeric_variable - model_variable) ** 2

    def binary_hamming(self, model, numeric_variable, model_variable):
        return numeric_variable * model_variable + (1 - numeric_variable) * model_variable

    def binary_crossentropy(self, model, numeric_variable, model_variable):
        return -(model_variable * np.log(numeric_variable) + (1 - model_variable) * np.log(1 - numeric_variable))

    def swapped_binary_crossentropy(self, model, numeric_variable, model_variable):
        _log = self.log_fn(model)
        return -(numeric_variable * _log(model_variable) + (1 - numeric_variable) * model.log(model_variable))

    def categorical_hamming(self, model, numeric_variable, model_variable):
        return 1 - model_variable[numeric_variable]

    def categorical_crossentropy(self, model, numeric_variable, model_variable):
        _sum = self.sum_fn(model)
        return -sum(model_variable * np.log(numeric_variable))

    def __call__(self, loss, model, numeric_variables, model_variables, sample_weight):
        # use uniform weights if none are passed, otherwise normalize the weights so that they sum to len(samples)
        if sample_weight is None:
            sample_weight = np.ones(len(numeric_variables))
        else:
            sample_weight = len(sample_weight) * np.array(sample_weight) / np.sum(sample_weight)
        # compute sum of losses
        _sum = self.sum_fn(model)
        return _sum([sw * loss(model, nv, mv) for nv, mv, sw in zip(numeric_variables, model_variables, sample_weight)])
