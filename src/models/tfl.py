"""Basic Lattice Model with Tensorflow-Lattice."""
from typing import List, Optional, Callable

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_lattice as tfl


class ColumnInfo:
    """

    Args:
        name: the column name.
        kind: the column kind, either 'numeric' (default) or 'categorical'.
        **kwargs: arguments for the `FeatureConfig` instance.
    """

    def __init__(self, name: str, kind: str = 'numeric', **kwargs):
        self.name: str = name
        self.kind: str = kind
        self.config: tfl.configs.FeatureConfig = tfl.configs.FeatureConfig(name=name, **kwargs)


class TFL:
    """A Lattice model built with Tensorflow-Lattice.

    Args:
        head: the estimator kind, either 'regression', 'binary' or 'multiclass'.
        columns: a list of `ColumnInfo` to build the calibration layers.
        pre_processing: routine for pre-processing.
        post_processing: routine for post-processing.
    """

    @staticmethod
    def input_fn(x: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> Callable:
        """Builds the inputs to be passed to the lattice model.

        Args:
            x: the input features.
            y: the labels.
            **kwargs: custom arguments.

        Returns:
            A callable function.
        """
        return tf.compat.v1.estimator.inputs.pandas_input_fn(x=x, y=y, shuffle=False, **kwargs)

    def __init__(self,
                 head: str,
                 columns: List[ColumnInfo],
                 pre_processing: Optional[Callable] = None,
                 post_processing: Optional[Callable] = None):
        if head == 'regression':
            self.model_kind = tfl.estimators.CannedRegressor
        elif head in ['binary', 'multiclass']:
            self.model_kind = tfl.estimators.CannedClassifier
        else:
            raise ValueError(f"'{head}' is not a supported head.")
        self.head: str = head
        self.columns: List[ColumnInfo] = columns
        self.pre_processing: Callable = (lambda x, y: (x, y)) if pre_processing is None else pre_processing
        self.post_processing: Callable = (lambda y: y) if post_processing is None else post_processing
        self.model: Optional[tfl.estimators.CannedEstimator] = None

    def build(self, x: pd.DataFrame, y: pd.Series, optimizer: str = 'Adam', seed: int = 0, **kwargs):
        """Builds the model on the given data.

        Args:
            x: the input features.
            y: the labels.
            optimizer: the optimizer.
            seed: the tensorflow random seed.
            **kwargs: custom arguments to be passed to the input_fn function.
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
        self.model = self.model_kind(
            feature_columns=feature_columns,
            model_config=model_config,
            feature_analysis_input_fn=TFL.input_fn(x=x, y=y, **kwargs),
            optimizer=optimizer,
            config=tf.estimator.RunConfig(tf_random_seed=seed)
        )

    def fit(self, x: pd.DataFrame, y: pd.Series, epochs: int, **kwargs):
        """Fits the model for the given number of epochs.

        Args:
            x: the input features.
            y: the labels.
            epochs: the number of epochs.
            **kwargs: custom arguments to be passed to the input_fn function.
        """
        assert self.model is not None, "model has not been built yet, please call model.build()"
        x, y = self.pre_processing(x, y)
        self.model.train(input_fn=TFL.input_fn(x=x, y=y, num_epochs=epochs, **kwargs))

    def predict(self, x: pd.DataFrame, **kwargs) -> np.ndarray:
        """Returns the predictions from the previously fit model.

        Args:
            x: the input features.
            **kwargs: custom arguments to be passed to the input_fn function.

        Returns:
            The predictions.
        """
        assert self.model is not None, "model has not been built yet, please call model.build()"
        x, _ = self.pre_processing(x, 0.0)
        outputs = self.model.predict(input_fn=TFL.input_fn(x=x, **kwargs))
        if self.head == 'regression':
            predictions = [o['predictions'] for o in outputs]
        elif self.head == 'binary':
            predictions = [o['logistic'] for o in outputs]
        elif self.head == 'multiclass':
            predictions = [o['probabilities'] for o in outputs]
        else:
            raise ValueError(f"'{self.head}' is not a supported head.")
        return self.post_processing(np.array(predictions))
