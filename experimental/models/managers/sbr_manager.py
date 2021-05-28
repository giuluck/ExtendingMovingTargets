"""Semantic-Based Regularizer Manager."""

import numpy as np
from typing import Any, Callable, Optional
from tensorflow.python.keras.callbacks import EarlyStopping

from experimental.models.managers import MLPManager
from src.models import SBR, SBRBatchGenerator, UnivariateSBR
from experimental.datasets.managers import TestManager, Fold
from experimental.models.managers.model_manager import ModelManager


# noinspection PyMissingOrEmptyDocstring
class SBRManager(ModelManager):
    def __init__(self,
                 test_manager: TestManager,
                 alpha: Optional[float] = None,
                 regularizer_act: Optional[Callable] = None,
                 epochs: int = 1000,
                 validation_split: float = 0.2,
                 early_stop: EarlyStopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
                 verbose: bool = False,
                 **kwargs):
        super(SBRManager, self).__init__(test_manager=test_manager,
                                         epochs=epochs,
                                         callbacks=[early_stop],
                                         verbose=verbose,
                                         **kwargs)
        self.validation_split = validation_split
        self.alpha = alpha
        self.regularizer_act = regularizer_act

    def fit(self, fold: Fold) -> Any:
        # handle validation set manually by removing part of the original data (plus the respective augmented samples)
        x, y, label = fold.x, fold.y, self.test_manager.dataset.y_column
        validation_indices = np.array(y[~np.isnan(y[label])].sample(frac=self.validation_split).index)
        val_data = (x.iloc[validation_indices], y[label].iloc[validation_indices])
        train_mask = ~np.isin(y['ground_index'], y['ground_index'].iloc[validation_indices])
        x, y = x[train_mask], y[train_mask]
        # handle batches and training
        sbr_batches = SBRBatchGenerator(x, y[label], y['ground_index'], y['monotonicity'], 4)
        model = SBR(output_act=self.output_act, h_units=self.h_units, alpha=self.alpha,
                    regularizer_act=self.regularizer_act, scalers=fold.scalers)
        model.compile(loss=self.loss, optimizer=self.optimizer)
        model.fit(sbr_batches, validation_data=val_data, **self.fit_info)
        return model


# noinspection PyMissingOrEmptyDocstring
class UnivariateSBRManager(MLPManager):
    def __init__(self,
                 test_manager: TestManager,
                 alpha: Optional[float] = None,
                 regularizer_act: Optional[Callable] = None,
                 direction: int = 1,
                 epochs: int = 1000,
                 validation_split: float = 0.2,
                 early_stop: EarlyStopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
                 verbose: bool = False,
                 **kwargs):
        super(UnivariateSBRManager, self).__init__(test_manager=test_manager,
                                                   epochs=epochs,
                                                   validation_split=validation_split,
                                                   early_stop=early_stop,
                                                   verbose=verbose,
                                                   **kwargs)
        self.alpha = alpha
        self.regularizer_act = regularizer_act
        self.direction = direction

    def fit(self, fold: Fold) -> Any:
        model = UnivariateSBR(output_act=self.output_act, h_units=self.h_units, alpha=self.alpha, input_dim=1,
                              regularizer_act=self.regularizer_act, direction=self.direction, scalers=fold.scalers)
        model.compile(loss=self.loss, optimizer=self.optimizer)
        model.fit(x=fold.x, y=fold.y, **self.fit_info)
        return model
