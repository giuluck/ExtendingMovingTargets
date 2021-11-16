"""Basic Lattice Model with Tensorflow-Lattice."""
from typing import List, Optional, Callable

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_lattice as tfl


class ColumnInfo:
    """Keeps the information about a `TFL` column."""

    def __init__(self, name: str, kind: str = 'numeric', **feature_config_kwargs):
        """
        :param name:
            The column name.

        :param kind:
            The column kind, either 'numeric' (default) or 'categorical'.

        :param feature_config_kwargs:
            Arguments for the `tfl.configs.FeatureConfig` instance.
        """
        self.name: str = name
        self.kind: str = kind
        self.config: tfl.configs.FeatureConfig = tfl.configs.FeatureConfig(name=name, **feature_config_kwargs)


class TFL:
    """A Lattice model built with Tensorflow-Lattice."""

    @staticmethod
    def input_fn(x: pd.DataFrame, y: Optional[pd.Series] = None, **input_fn_kwargs) -> Callable:
        """Builds the inputs to be passed to the lattice model.

        :param x:
            The input features.

        :param y:
            The labels.

        :param input_fn_kwargs:
            Arguments for the method `tf.compat.v1.estimator.inputs.pandas_input_fn()` (e.g., batch size, num_epochs).

        :returns:
            A callable function that processes the input data.
        """
        return tf.compat.v1.estimator.inputs.pandas_input_fn(x=x, y=y, shuffle=False, **input_fn_kwargs)

    def __init__(self,
                 head: str,
                 columns: List[ColumnInfo],
                 pre_processing: Optional[Callable] = None,
                 post_processing: Optional[Callable] = None):
        """
        :param head:
            The estimator kind, either 'regression', 'binary' or 'multiclass'.

        :param columns:
            A list of `ColumnInfo` to build the calibration layers.

        :param pre_processing:
            Routine for pre-processing.

        :param post_processing:
            Routine for post-processing.
        """

        self.head: str = head
        """The estimator kind, either 'regression', 'binary' or 'multiclass'."""

        self.model_kind = None
        """The estimator kind, either 'CannedRegressor' or 'CannedClassifier' depending on the head."""

        self.columns: List[ColumnInfo] = columns
        """A list of `ColumnInfo` to build the calibration layers."""

        self.pre_processing: Callable = (lambda x, y: (x, y)) if pre_processing is None else pre_processing
        """Routine for pre-processing."""

        self.post_processing: Callable = (lambda y: y) if post_processing is None else post_processing
        """Routine for post-processing."""

        self._model: Optional[tfl.estimators.CannedEstimator] = None
        """The inner TFL Canned Estimator."""

        # handles model kind depending on the chosen head
        if head == 'regression':
            self.model_kind = tfl.estimators.CannedRegressor
        elif head in ['binary', 'multiclass']:
            self.model_kind = tfl.estimators.CannedClassifier
        else:
            raise ValueError(f"'{head}' is not a supported head.")

    def build(self, x: pd.DataFrame, y: pd.Series, optimizer: str = 'Adam', seed: int = 0, **input_fn_kwargs):
        """Builds the model on the given data.

        :param x:
            The input features.

        :param y:
            The ground truths.

        :param optimizer:
            The optimizer.

        :param seed:
            The tensorflow random seed.

        :param input_fn_kwargs:
            Arguments for the `input_fn` function, which eventually passes them to the method
            `tf.compat.v1.estimator.inputs.pandas_input_fn()` (e.g., batch_size, num_epochs, ...)
        """
        feature_columns = []
        for column in self.columns:
            if column.kind == 'numeric':
                feature = tf.feature_column.numeric_column(column.name)
            elif column.kind == 'categorical':
                feature = tf.feature_column.categorical_column_with_vocabulary_list(
                    column.name,
                    vocabulary_list=x[column.name].unique(),
                    dtype=tf.string,
                    default_value=0
                )
            else:
                raise ValueError(f"'{column.kind}' is not a supported column kind.")
            feature_columns.append(feature)
        model_config = tfl.configs.CalibratedLatticeConfig(feature_configs=[c.config for c in self.columns])
        self._model = self.model_kind(
            feature_columns=feature_columns,
            model_config=model_config,
            feature_analysis_input_fn=TFL.input_fn(x=x, y=y, **input_fn_kwargs),
            optimizer=optimizer,
            config=tf.estimator.RunConfig(tf_random_seed=seed)
        )

    def fit(self, x: pd.DataFrame, y: pd.Series, epochs: int, **input_fn_kwargs):
        """Fits the model for the given number of epochs.

        :param x:
            The input features.

        :param y:
            The ground truths.

        :param epochs:
            The number of epochs.

        :param input_fn_kwargs:
            Arguments for the `input_fn` function, which eventually passes them to the method
            `tf.compat.v1.estimator.inputs.pandas_input_fn()` (e.g., batch_size, num_epochs, ...)
        """
        assert self._model is not None, "model has not been built yet, please call model.build()"
        x, y = self.pre_processing(x, y)
        self._model.train(input_fn=TFL.input_fn(x=x, y=y, num_epochs=epochs, **input_fn_kwargs))

    def predict(self, x: pd.DataFrame, **input_fn_kwargs) -> np.ndarray:
        """Returns the predictions from the previously fit model.

        :param x:
            The input features.

        :param input_fn_kwargs:
            Arguments for the `input_fn` function, which eventually passes them to the method
            `tf.compat.v1.estimator.inputs.pandas_input_fn()` (e.g., batch_size, num_epochs, ...)

        :returns:
            The vector of predictions.
        """
        assert self._model is not None, "model has not been built yet, please call model.build()"
        x, _ = self.pre_processing(x, None)
        outputs = self._model.predict(input_fn=TFL.input_fn(x=x, **input_fn_kwargs))
        if self.head == 'regression':
            predictions = [o['predictions'] for o in outputs]
        elif self.head == 'binary':
            predictions = [o['logistic'] for o in outputs]
        elif self.head == 'multiclass':
            predictions = [o['probabilities'] for o in outputs]
        else:
            raise ValueError(f"'{self.head}' is not a supported head.")
        return self.post_processing(np.array(predictions))
