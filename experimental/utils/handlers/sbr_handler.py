"""Semantic-Based Regularizer Handler."""

from typing import Any, Callable, Optional, Union

import numpy as np
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

from experimental.utils.handlers.mlp_handler import MLPHandler, Fold
from src.datasets import AbstractManager
from src.models import SBR, SBRBatchGenerator, UnivariateSBR
from src.util.typing import Augmented


# noinspection PyMissingOrEmptyDocstring
class SBRHandler(MLPHandler):
    def __init__(self,
                 manager: AbstractManager,
                 loss: str,
                 alpha: Union[None, float, OptimizerV2] = None,
                 regularizer_act: Optional[Callable] = None,
                 num_augmented: Optional[Augmented] = None,
                 num_random: int = 0,
                 num_ground: Optional[int] = None,
                 **kwargs):
        super(SBRHandler, self).__init__(manager=manager, loss=loss, **kwargs)
        self.alpha: Union[None, float, OptimizerV2] = alpha
        self.regularizer_act: Optional[Callable] = regularizer_act
        self.num_augmented: Optional[Augmented] = num_augmented
        self.num_random: int = num_random
        self.num_ground: Optional[int] = num_ground

    def fit(self, fold: Fold) -> Any:
        # handle validation set manually by removing part of the original data
        x, y, label = fold.x.reset_index(drop=True), fold.y.reset_index(drop=True), self.manager.label
        val_mask = np.isin(np.arange(len(y)), y.sample(frac=self.validation_split).index.values)
        val_data = (x[val_mask], y[val_mask])
        (x, y), scalers = self.manager.get_augmented_data(x=x[~val_mask],
                                                          y=y[~val_mask],
                                                          monotonicities=True,
                                                          num_augmented=self.num_augmented,
                                                          num_random=self.num_random,
                                                          num_ground=self.num_ground)
        # handle batches and training
        sbr_batches = SBRBatchGenerator(x=x, y=y[label], ground_indices=y['ground_index'],
                                        monotonicities=y['monotonicity'], batch_size=self.batch_size)
        model = SBR(output_act=self.output_act, h_units=self.h_units, alpha=self.alpha,
                    regularizer_act=self.regularizer_act, scalers=fold.scalers)
        model.compile(loss=self.loss, optimizer=self.optimizer, run_eagerly=self.run_eagerly)
        model.fit(sbr_batches, validation_data=val_data, epochs=self.epochs, batch_size=self.batch_size,
                  callbacks=self.callbacks, verbose=self.verbose)
        return model


# noinspection PyMissingOrEmptyDocstring
class UnivariateSBRHandler(MLPHandler):
    def __init__(self,
                 manager: AbstractManager,
                 loss: str,
                 direction: int,
                 alpha: Union[None, float, OptimizerV2] = None,
                 regularizer_act: Optional[Callable] = None,
                 **kwargs):
        super(UnivariateSBRHandler, self).__init__(manager=manager, loss=loss, **kwargs)
        self.direction: int = direction
        self.alpha: Union[None, float, OptimizerV2] = alpha
        self.regularizer_act: Optional[Callable] = regularizer_act

    def fit(self, fold: Fold) -> Any:
        model = UnivariateSBR(output_act=self.output_act, h_units=self.h_units, alpha=self.alpha, input_dim=1,
                              regularizer_act=self.regularizer_act, direction=self.direction, scalers=fold.scalers)
        model.compile(loss=self.loss, optimizer=self.optimizer, run_eagerly=self.run_eagerly)
        model.fit(x=fold.x, y=fold.y, validation_split=self.validation_split, epochs=self.epochs,
                  batch_size=self.batch_size, callbacks=self.callbacks, verbose=self.verbose)
        return model
