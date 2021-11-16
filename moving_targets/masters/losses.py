"""Losses utilities."""

from typing import Callable, Optional, Any

import numpy as np

from moving_targets.util.typing import Vector, Number


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

    def __call__(self, model, numeric_variables: Vector, model_variables: Vector,
                 sample_weight: Optional[Vector] = None) -> float:
        """Computes the aggregated sum of losses over a paired set of vectors, a true and a predicted one.

        :param model:
            The optimization model.

        :param numeric_variables:
            The real-number variables (i.e., the ground truths).

        :param model_variables:
            The model variables (i.e., the predictions).

        :param sample_weight:
            The sample weights associated to each sample.

        :returns:
            A real number representing the final loss.
        """
        # use uniform weights if none are passed, otherwise normalize the weights so that they sum to len(samples)
        if sample_weight is None:
            sample_weight = np.ones(len(numeric_variables))
        else:
            sample_weight = len(sample_weight) * np.array(sample_weight) / np.sum(sample_weight)
        # compute sum of losses
        variables = zip(numeric_variables, model_variables, sample_weight)
        return self._sum_fn(model, [sw * self._loss_fn(model, nv, mv) for nv, mv, sw in variables])


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

    def __call__(self, model, numeric_variables: Vector, model_variables: Vector,
                 sample_weight: Optional[Vector] = None) -> float:
        """Computes the aggregated sum of losses over a paired set of vectors, a true and a predicted one.

        :param model:
            The optimization model.

        :param numeric_variables:
            The real-number variables (i.e., the ground truths).

        :param model_variables:
            The model variables (i.e., the predictions).

        :param sample_weight:
            The sample weights associated to each sample.

        :returns:
            A real number representing the final loss.
        """
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

    def __call__(self, model, numeric_variables: Vector, model_variables: Vector,
                 sample_weight: Optional[Vector] = None) -> float:
        """Computes the aggregated sum of losses over a paired set of vectors, a true and a predicted one.

        :param model:
            The optimization model.

        :param numeric_variables:
            The real-number variables (i.e., the ground truths).

        :param model_variables:
            The model variables (i.e., the predictions).

        :param sample_weight:
            The sample weights associated to each sample.

        :returns:
            A real number representing the final loss.
        """
        int_numeric_variables: Vector = np.array(numeric_variables).astype(int)
        assert np.allclose(numeric_variables, int_numeric_variables), f'cannot use as index values {numeric_variables}'
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

    def __call__(self, model, numeric_variables: Vector, model_variables: Vector,
                 sample_weight: Optional[Vector] = None) -> float:
        """Computes the aggregated sum of losses over a paired set of vectors, a true and a predicted one.

        :param model:
            The optimization model.

        :param numeric_variables:
            The real-number variables (i.e., the ground truths).

        :param model_variables:
            The model variables (i.e., the predictions).

        :param sample_weight:
            The sample weights associated to each sample.

        :returns:
            A real number representing the final loss.
        """
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
                 sum_fn: Callable = lambda model, x: model.sum(x),
                 abs_fn: Callable = lambda model, x: model.abs(x),
                 log_fn: Callable = lambda model, x: model.log(x)):
        """
        :param sum_fn:
            Routine function that computes the sum of a vector x using the given model.

        :param abs_fn:
            Routine function that computes the absolute value of a vector x using the given model.

        :param log_fn:
            Routine function that computes the logarithm of a vector x using the given model.
        """

        self._sum_fn: Callable = sum_fn
        """Routine function that computes the sum of a vector x using the given model."""

        self._abs_fn: Callable = abs_fn
        """Routine function that computes the absolute value of a vector x using the given model."""

        self._log_fn: Callable = log_fn
        """Routine function that computes the logarithm of a vector x using the given model."""

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

    def _absolute_errors(self, model, numeric_variable: Number, model_variable: Any):
        """Computes the absolute error loss using the custom `abs_fn` function.

        :param model:
            The solver instance.

        :param numeric_variable:
            The numpy variables.

        :param model_variable:
            The solver variables.

        :returns:
            The expression of the absolute error loss as created by the solver.
        """
        lb = numeric_variable - model_variable.ub
        ub = numeric_variable - model_variable.lb
        return self._abs_fn(model, numeric_variable - model_variable, lb=lb, ub=ub)

    # noinspection PyMethodMayBeStatic, PyUnusedLocal
    def _squared_errors(self, model, numeric_variable: Number, model_variable: Any):
        """Computes the squared error loss.

        :param model:
            The solver instance (unused).

        :param numeric_variable:
            The numpy variables.

        :param model_variable:
            The solver variables.

        :returns:
            The expression of the squared error loss as created by the solver.
        """
        return (numeric_variable - model_variable) ** 2

    def _binary_hamming(self, model, numeric_variable: int, model_variable: Any):
        """Computes the binary hamming distance.

        :param model:
            The solver instance.

        :param numeric_variable:
            The numpy variables.

        :param model_variable:
            The solver variables.

        :returns:
            The expression of the binary hamming distance as created by the solver.
        """
        # if model_variable is 0 then the array becomes [1, 0], otherwise if it is 1 then the array becomes [0, 1]
        model_variable = np.array([1 - model_variable, model_variable])
        return self._categorical_hamming(model, numeric_variable, model_variable)

    def _binary_crossentropy(self, model, numeric_variable: Number, model_variable: Any):
        """Computes the binary crossentropy loss, obtained from the categorical formulation.

        :param model:
            The solver instance.

        :param numeric_variable:
            The numpy variables.

        :param model_variable:
            The solver variables.

        :returns:
            The expression of the binary crossentropy loss as created by the solver.
        """
        # if model_variable is 0 then the array becomes [1, 0], otherwise if it is 1 then the array becomes [0, 1]
        # if numeric_variable is p, then the probability of 0 is 1 - p and the probability of 1 is p,
        # thus the array becomes [1 - p, p]
        model_variable = np.array([1 - model_variable, model_variable])
        numeric_variable = np.array([1 - numeric_variable, numeric_variable])
        return self._categorical_crossentropy(model, numeric_variable, model_variable)

    def _reversed_binary_crossentropy(self, model, numeric_variable: Number, model_variable: Any):
        """Computes the reversed binary crossentropy loss using the custom `log_fn` function.

        :param model:
            The solver instance.

        :param numeric_variable:
            The numpy variables.

        :param model_variable:
            The solver variables.

        :returns:
            The expression of the reversed binary crossentropy loss as created by the solver.
        """
        # do not leverage _reversed_categorical_crossentropy to avoid Gurobi error "LinExpr has no attribute 'lb'"
        nv, mv = numeric_variable, model_variable
        l0 = (1 - nv) * self._log_fn(model, 1 - mv, lb=1 - mv.ub, ub=1 - mv.lb)  # loss w.r.t. term 0
        l1 = nv * self._log_fn(model, mv, lb=mv.lb, ub=mv.ub)  # loss w.r.t. term 1
        return -(l0 + l1)

    def _symmetric_binary_crossentropy(self, model, numeric_variable: Number, model_variable: Any):
        """Computes the symmetric binary crossentropy loss, obtained as the sum of the standard and reversed ones.

        :param model:
            The solver instance.

        :param numeric_variable:
            The numpy variables.

        :param model_variable:
            The solver variables.

        :returns:
            The expression of the symmetric binary crossentropy loss as created by the solver.
        """
        bce = self._binary_crossentropy(model, numeric_variable, model_variable)
        rbce = self._reversed_binary_crossentropy(model, numeric_variable, model_variable)
        return bce + rbce

    # noinspection PyMethodMayBeStatic, PyUnusedLocal
    def _categorical_hamming(self, model, numeric_variable: int, model_variable: Vector):
        """Computes the categorical hamming distance.

        :param model:
            The solver instance.

        :param numeric_variable:
            The numpy variables.

        :param model_variable:
            The solver variables.

        :returns:
            The expression of the categorical hamming distance as created by the solver.
        """
        return 1 - model_variable[numeric_variable]

    def _categorical_crossentropy(self, model, numeric_variable: Vector, model_variable: Vector):
        """Computes the categorical crossentropy loss.

        :param model:
            The solver instance.

        :param numeric_variable:
            The numpy variables.

        :param model_variable:
            The solver variables.

        :returns:
            The expression of the categorical crossentropy loss as created by the solver.
        """
        return self._sum_fn(model, -model_variable * np.log(numeric_variable))

    def _reversed_categorical_crossentropy(self, model, numeric_variable: Vector, model_variable: Vector):
        """Computes the reversed categorical crossentropy loss using the custom `log_fn` function.

        :param model:
            The solver instance.

        :param numeric_variable:
            The numpy variables.

        :param model_variable:
            The solver variables.

        :returns:
            The expression of the reversed categorical crossentropy loss as created by the solver.
        """

        def _log(x):
            """Support function to compute the logarithm using the custom `log_fn` function"""
            return self._log_fn(model, x, lb=x.lb, ub=x.ub)

        return self._sum_fn(model, [-nv * _log(mv) for nv, mv in zip(numeric_variable, model_variable)])

    def _symmetric_categorical_crossentropy(self, model, numeric_variable: Vector, model_variable: Any):
        """Computes the symmetric categorical crossentropy loss, obtained as the sum of the standard and reversed ones.

        :param model:
            The solver instance.

        :param numeric_variable:
            The numpy variables.

        :param model_variable:
            The solver variables.

        :returns:
            The expression of the symmetric categorical crossentropy loss as created by the solver.
        """
        cce = self._categorical_crossentropy(model, numeric_variable, model_variable)
        rcce = self._reversed_categorical_crossentropy(model, numeric_variable, model_variable)
        return cce + rcce
