"""Losses utilities."""

from typing import Callable, Optional

import numpy as np


class SumLoss:
    """Callable class that computes the losses over a set of pairs, then aggregates with a sum."""

    def __init__(self, loss_fn: Callable, sum_fn: Callable):
        """
        :param loss_fn:
            Routine function that computes the loss over a pair of values, a true and a predicted one.

        :param sum_fn:
            Routine function that computes the sum of a vector x using the given model.
        """

        self._loss_fn: Callable = loss_fn
        """Routine function that computes the loss over a pair of values, a true and a predicted one."""

        self._sum_fn: Callable = sum_fn
        """Routine function that computes the sum of a vector x using the given model."""

    def __call__(self, model, numeric_variables, model_variables, sample_weight: Optional = None) -> float:
        """Computes the aggregated sum of losses over a paired set of vectors, a true and a predicted one.

        :param model:
            The optimization model.

        :param numeric_variables:
            The real-number variables (i.e., the ground truths).

        :param model_variables:
            The model variables (i.e., the predictions).

        :param sample_weight:
            The sample weights associated to each sample.

        :return:
            A real number representing the final loss.
        """
        # use uniform weights if none are passed, otherwise normalize the weights so that they sum to len(samples)
        if sample_weight is None:
            sample_weight = np.ones(len(numeric_variables))
        else:
            sample_weight = len(sample_weight) * np.array(sample_weight) / np.sum(sample_weight)
        # compute sum of returned losses
        partial_losses = self._loss_fn(model, numeric_variables, model_variables, sample_weight)
        return self._sum_fn(model, partial_losses)


class MeanLoss(SumLoss):
    """Callable class that computes the losses over a set of pairs, then aggregates with an average."""

    def __init__(self, loss_fn: Callable, sum_fn: Callable):
        """
        :param loss_fn:
            Routine function that computes the loss over a pair of values, a true and a predicted one.

        :param sum_fn:
            Routine function that computes the sum of a vector x using the given model.
        """
        super(MeanLoss, self).__init__(loss_fn=loss_fn, sum_fn=sum_fn)

    def __call__(self, model, numeric_variables, model_variables, sample_weight: Optional = None) -> float:
        return super(MeanLoss, self).__call__(
            model=model,
            numeric_variables=numeric_variables,
            model_variables=model_variables,
            sample_weight=sample_weight
        ) / len(numeric_variables)


class IntMeanLoss(MeanLoss):
    """Callable class that computes a mean loss over a set of pairs, the first one converted to integer values."""

    def __init__(self, loss_fn: Callable, sum_fn: Callable):
        """
        :param loss_fn:
            Routine function that computes the loss over a pair of values, a true and a predicted one.

        :param sum_fn:
            Routine function that computes the sum of a vector x using the given model.
        """
        super(IntMeanLoss, self).__init__(loss_fn=loss_fn, sum_fn=sum_fn)

    def __call__(self, model, numeric_variables, model_variables, sample_weight: Optional = None) -> float:
        int_numeric_variables = np.array(numeric_variables).astype(int)
        assert np.allclose(numeric_variables, int_numeric_variables), 'the given numeric variables are not integer'
        return super(IntMeanLoss, self).__call__(
            model=model,
            numeric_variables=int_numeric_variables,
            model_variables=model_variables,
            sample_weight=sample_weight
        )


class ClippedMeanLoss(MeanLoss):
    """Callable class that computes the losses over a set of pairs, then aggregates with an average once the real
       variables have been clipped within the interval [clip_value, 1 - clip_value]."""

    def __init__(self, loss_fn: Callable, sum_fn: Callable, clip_value: float = 1e-15):
        """
        :param loss_fn:
            Routine function that computes the loss over a pair of values, a true and a predicted one.

        :param sum_fn:
            Routine function that computes the sum of a vector x using the given model.

        :param clip_value:
            Floating point value that indicates the width of the interval.
        """
        super(ClippedMeanLoss, self).__init__(loss_fn=loss_fn, sum_fn=sum_fn)

        self._clip_value: float = clip_value
        """Floating point value that indicates the width of the interval."""

    def __call__(self, model, numeric_variables, model_variables, sample_weight: Optional = None) -> float:
        numeric_variables = np.clip(numeric_variables.astype(float), a_min=self._clip_value, a_max=1 - self._clip_value)
        return super(ClippedMeanLoss, self).__call__(
            model=model,
            numeric_variables=numeric_variables,
            model_variables=model_variables,
            sample_weight=sample_weight
        )


class LossesHandler:
    """Util class that handles the functions for the principal losses to be used in the Master."""

    def __init__(self,
                 sum_fn: Optional[Callable] = lambda model, vector: np.sum(vector),
                 sqr_fn: Optional[Callable] = lambda model, vector: vector**2,
                 abs_fn: Optional[Callable] = lambda model, vector: np.abs(vector),
                 log_fn: Optional[Callable] = lambda model, vector: np.log(vector)):
        """
        :param sum_fn:
            Routine function of type f(<model>, <vector>) -> <float> that computes the sum of variables.

        :param sqr_fn:
            Routine function of type f(<model>, <vector>) -> <vector> that computes the squared value of each element.

        :param abs_fn:
            Routine function of type f(<model>, <vector>) -> <vector> that computes the absolute value of each element.

        :param log_fn:
            Routine function of type f(<model>, <vector>) -> <vector> that computes the logarithm of each element.
        """

        def _fn(operation: str):
            raise ValueError(f'This solver cannot deal with {operation}.')

        self._sum_fn: Callable = sum_fn if sum_fn is not None else lambda model, vector: _fn(operation='sums')
        """Routine function that computes the sum of variables."""

        self._sqr_fn: Callable = sqr_fn if sqr_fn is not None else lambda model, vector: _fn(operation='squared values')
        """Routine function that computes each partial squared loss."""

        self._abs_fn: Callable = abs_fn if abs_fn is not None else lambda model, vector: _fn(operation='abs values')
        """Routine function that computes each partial absolute loss."""

        self._log_fn: Callable = log_fn if log_fn is not None else lambda model, vector: _fn(operation='logarithms')
        """Routine function that computes each partial log loss."""

        self.sum_of_absolute_errors: SumLoss = SumLoss(loss_fn=self._absolute_errors, sum_fn=self._sum_fn)
        """`SumLoss` object that computes the sum of absolute errors loss."""

        self.sum_of_squared_errors: SumLoss = SumLoss(loss_fn=self._squared_errors, sum_fn=self._sum_fn)
        """`SumLoss` object that computes the sum of squared errors loss."""

        self.mean_absolute_error: MeanLoss = MeanLoss(loss_fn=self._absolute_errors, sum_fn=self._sum_fn)
        """`MeanLoss` object that computes the mean absolute error loss."""

        self.mean_squared_error: MeanLoss = MeanLoss(loss_fn=self._squared_errors, sum_fn=self._sum_fn)
        """`MeanLoss` object that computes the mean squared error loss."""

        self.binary_hamming: IntMeanLoss = IntMeanLoss(loss_fn=self._binary_hamming, sum_fn=self._sum_fn)
        """`IntMeanLoss` object that computes the binary hamming distance."""

        self.binary_crossentropy: ClippedMeanLoss = ClippedMeanLoss(
            loss_fn=self._binary_crossentropy,
            sum_fn=self._sum_fn
        )
        """`ClippedMeanLoss` object that computes the binary crossentropy loss."""

        self.reversed_binary_crossentropy: MeanLoss = MeanLoss(
            loss_fn=self._reversed_binary_crossentropy,
            sum_fn=self._sum_fn
        )
        """`MeanLoss` object that computes the reversed binary crossentropy loss."""

        self.symmetric_binary_crossentropy: ClippedMeanLoss = ClippedMeanLoss(
            loss_fn=self._symmetric_binary_crossentropy,
            sum_fn=self._sum_fn
        )
        """`ClippedMeanLoss` object that computes the symmetric binary crossentropy loss."""

        self.categorical_hamming: IntMeanLoss = IntMeanLoss(
            loss_fn=self._categorical_hamming,
            sum_fn=self._sum_fn
        )
        """`IntMeanLoss` object that computes the categorical hamming distance."""

        self.categorical_crossentropy: ClippedMeanLoss = ClippedMeanLoss(
            loss_fn=self._categorical_crossentropy,
            sum_fn=self._sum_fn
        )
        """`ClippedMeanLoss` object that computes the categorical crossentropy loss."""

        self.reversed_categorical_crossentropy: MeanLoss = MeanLoss(
            loss_fn=self._reversed_categorical_crossentropy,
            sum_fn=self._sum_fn
        )
        """`MeanLoss` object that computes the reversed categorical crossentropy loss."""

        self.symmetric_categorical_crossentropy: ClippedMeanLoss = ClippedMeanLoss(
            loss_fn=self._symmetric_categorical_crossentropy,
            sum_fn=self._sum_fn
        )
        """`ClippedMeanLoss` object that computes the symmetric categorical crossentropy loss."""

    def _absolute_errors(self, model, numeric_variables, model_variables, sample_weights):
        # transpose absolute errors array in order to correctly deal with multivariate targets
        absolute_errors = self._abs_fn(model, model_variables - numeric_variables)
        return sample_weights * absolute_errors.transpose()

    # noinspection PyMethodMayBeStatic, PyUnusedLocal
    def _squared_errors(self, model, numeric_variables, model_variables, sample_weights):
        # transpose squared errors array in order to correctly deal with multivariate targets
        squared_errors = self._sqr_fn(model, model_variables - numeric_variables)
        return sample_weights * squared_errors.transpose()

    def _binary_hamming(self, model, numeric_variables, model_variables, sample_weights):
        # if model_variable is 0 then the array becomes [1, 0], otherwise if it is 1 then the array becomes [0, 1]
        model_variables = np.array([1 - model_variables, model_variables]).transpose()
        return self._categorical_hamming(model, numeric_variables, model_variables, sample_weights)

    def _binary_crossentropy(self, model, numeric_variables, model_variables, sample_weights):
        # if model_variable is 0 then the array becomes [1, 0], otherwise if it is 1 then the array becomes [0, 1]
        # if numeric_variable is p, then the probability of 0 is 1 - p and the probability of 1 is p,
        # thus the array becomes [1 - p, p]
        model_variables = np.array([1 - model_variables, model_variables]).transpose()
        numeric_variables = np.array([1 - numeric_variables, numeric_variables]).transpose()
        return self._categorical_crossentropy(model, numeric_variables, model_variables, sample_weights)

    def _reversed_binary_crossentropy(self, model, numeric_variables, model_variables, sample_weights):
        # if model_variable is 0 then the array becomes [1, 0], otherwise if it is 1 then the array becomes [0, 1]
        # if numeric_variable is p, then the probability of 0 is 1 - p and the probability of 1 is p,
        # thus the array becomes [1 - p, p]
        model_variables = np.array([1 - model_variables, model_variables]).transpose()
        numeric_variables = np.array([1 - numeric_variables, numeric_variables]).transpose()
        return self._reversed_categorical_crossentropy(model, numeric_variables, model_variables, sample_weights)

    def _symmetric_binary_crossentropy(self, model, numeric_variables, model_variables, sample_weights):
        bce = self._binary_crossentropy(model, numeric_variables, model_variables, sample_weights)
        rbce = self._reversed_binary_crossentropy(model, numeric_variables, model_variables, sample_weights)
        return bce + rbce

    # noinspection PyMethodMayBeStatic, PyUnusedLocal
    def _categorical_hamming(self, model, numeric_variables, model_variables, sample_weights):
        # model_variables: (n, c), where c is the number of classes
        # numeric variables: (n, )
        #
        # in order to exploit numpy vector computation instead of list comprehensions, convert numeric variables (which
        # contain class indices, e.g., [2, 5, 0, 2, 1, ...]) into mask of indices for the flattened version of the
        # model variables, having shape (n * c, ); the indices are obtained by creating a range vector with n elements
        # and step c (i.e., [0, c, 2c, 3c, ...]) and adding the respective class index which is into numeric variables
        num_samples, num_classes = model_variables.shape
        indices_mask = np.arange(num_samples * num_classes, step=num_classes) + numeric_variables
        return sample_weights * (1 - model_variables.flatten()[indices_mask])

    def _categorical_crossentropy(self, model, numeric_variables, model_variables, sample_weights):
        log_losses = -model_variables * np.log(numeric_variables)
        return self._sum_fn(model, sample_weights * log_losses.transpose())

    def _reversed_categorical_crossentropy(self, model, numeric_variables, model_variables, sample_weights):
        shape = model_variables.shape
        log_losses = -numeric_variables * self._log_fn(model, model_variables.flatten()).reshape(shape)
        return self._sum_fn(model, sample_weights * log_losses.transpose())

    def _symmetric_categorical_crossentropy(self, model, numeric_variables, model_variables, sample_weights):
        cce = self._categorical_crossentropy(model, numeric_variables, model_variables, sample_weights)
        rcce = self._reversed_categorical_crossentropy(model, numeric_variables, model_variables, sample_weights)
        return cce + rcce
