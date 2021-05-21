"""Losses utilities."""

from typing import Callable, Optional

import numpy as np

from moving_targets.util.typing import Vector


class SumLoss:
    """Callable class that computes the losses over a set of pairs, then aggregates with a sum.

    Args:
        loss_fn: routine function that computes the loss over a pair of values, a true and a predicted one.
        sum_fn: routine function that computes the sum of a vector x using the given model.
    """

    def __init__(self, loss_fn: Callable, sum_fn: Callable):
        self.loss_fn: Callable = loss_fn
        self.sum_fn: Callable = sum_fn

    def __call__(self, model, numeric_variables: Vector, model_variables: Vector,
                 sample_weight: Optional[Vector] = None) -> float:
        """Computes the aggregated sum of losses over a paired set of vectors, a true and a predicted one.

        Args:
            model: the optimization model.
            numeric_variables: the real-number variables (i.e., the ground truths).
            model_variables: the model variables (i.e., the predictions).
            sample_weight: the sample weights associated to each sample.

        Returns:
            A real number representing the final loss.
        """
        # use uniform weights if none are passed, otherwise normalize the weights so that they sum to len(samples)
        if sample_weight is None:
            sample_weight = np.ones(len(numeric_variables))
        else:
            sample_weight = len(sample_weight) * np.array(sample_weight) / np.sum(sample_weight)
        # compute sum of losses
        variables = zip(numeric_variables, model_variables, sample_weight)
        return self.sum_fn(model, [sw * self.loss_fn(model, nv, mv) for nv, mv, sw in variables])


class MeanLoss(SumLoss):
    """Callable class that computes the losses over a set of pairs, then aggregates with an average.

    Args:
        loss_fn: routine function that computes the loss over a pair of values, a true and a predicted one.
        sum_fn: routine function that computes the sum of a vector x using the given model.
    """

    def __init__(self, loss_fn: Callable, sum_fn: Callable):
        super(MeanLoss, self).__init__(loss_fn=loss_fn, sum_fn=sum_fn)

    def __call__(self, model, numeric_variables: Vector, model_variables: Vector,
                 sample_weight: Optional[Vector] = None) -> float:
        return super(MeanLoss, self).__call__(
            model=model,
            numeric_variables=numeric_variables,
            model_variables=model_variables,
            sample_weight=sample_weight
        ) / len(numeric_variables)


class ClippedMeanLoss(MeanLoss):
    """Callable class that computes the losses over a set of pairs, then aggregates with an average once the real
       variables have been clipped within the interval [clip_value, 1 - clip_value].

    Args:
        loss_fn: routine function that computes the loss over a pair of values, a true and a predicted one.
        sum_fn: routine function that computes the sum of a vector x using the given model.
        clip_value: floating point value that indicates the width of the interval.
    """

    def __init__(self, loss_fn: Callable, sum_fn: Callable, clip_value: float = 1e-15):
        super(ClippedMeanLoss, self).__init__(loss_fn=loss_fn, sum_fn=sum_fn)
        self.clip_value: float = clip_value

    def __call__(self, model, numeric_variables: Vector, model_variables: Vector,
                 sample_weight: Optional[Vector] = None) -> float:
        numeric_variables = np.clip(numeric_variables, a_min=self.clip_value, a_max=1 - self.clip_value)
        return super(ClippedMeanLoss, self).__call__(
            model=model,
            numeric_variables=numeric_variables,
            model_variables=model_variables,
            sample_weight=sample_weight
        )


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class LossesHandler:
    """Util class that handles the functions for the principal losses to be used in the Master.

    Args:
        sum_fn: routine function that computes the sum of a vector x using the given model.
        abs_fn: routine function that computes the absolute value of a vector x using the given model.
        log_fn: routine function that computes the logarithm of a vector x using the given model.
    """

    def __init__(self,
                 sum_fn: Optional[Callable] = lambda model, x: model.sum(x),
                 abs_fn: Optional[Callable] = lambda model, x: model.abs(x),
                 log_fn: Optional[Callable] = lambda model, x: model.log(x)):
        # metadata
        self.sum_fn = sum_fn
        self.abs_fn = abs_fn
        self.log_fn = log_fn
        # losses
        self.sum_of_absolute_errors = SumLoss(loss_fn=self._absolute_errors, sum_fn=self.sum_fn)
        self.sum_of_squared_errors = SumLoss(loss_fn=self._squared_errors, sum_fn=self.sum_fn)
        self.mean_absolute_error = MeanLoss(loss_fn=self._absolute_errors, sum_fn=self.sum_fn)
        self.mean_squared_error = MeanLoss(loss_fn=self._squared_errors, sum_fn=self.sum_fn)
        self.binary_hamming = MeanLoss(loss_fn=self._binary_hamming, sum_fn=self.sum_fn)
        self.binary_crossentropy = ClippedMeanLoss(loss_fn=self._binary_crossentropy, sum_fn=self.sum_fn)
        self.swapped_binary_crossentropy = MeanLoss(loss_fn=self._swapped_binary_crossentropy, sum_fn=self.sum_fn)
        self.categorical_hamming = MeanLoss(loss_fn=self._categorical_hamming, sum_fn=self.sum_fn)
        self.categorical_crossentropy = ClippedMeanLoss(loss_fn=self._categorical_crossentropy, sum_fn=self.sum_fn)

    def _absolute_errors(self, model, numeric_variable, model_variable):
        return self.abs_fn(model, numeric_variable - model_variable)

    def _squared_errors(self, model, numeric_variable, model_variable):
        return (numeric_variable - model_variable) ** 2

    def _binary_hamming(self, model, numeric_variable, model_variable):
        return numeric_variable * model_variable + (1 - numeric_variable) * model_variable

    def _binary_crossentropy(self, model, numeric_variable, model_variable):
        return -(model_variable * np.log(numeric_variable) + (1 - model_variable) * np.log(1 - numeric_variable))

    def _swapped_binary_crossentropy(self, model, numeric_variable, model_variable):
        _log = lambda x: self.log_fn(model, x)
        return -(numeric_variable * _log(model_variable) + (1 - numeric_variable) * _log(1 - model_variable))

    def _categorical_hamming(self, model, numeric_variable, model_variable):
        return 1 - model_variable[numeric_variable]

    def _categorical_crossentropy(self, model, numeric_variable, model_variable):
        return self.sum_fn(model, -model_variable * np.log(numeric_variable))
