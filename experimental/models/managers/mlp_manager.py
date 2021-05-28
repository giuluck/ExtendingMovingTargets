"""Multi-Layer Perceptron Manager."""

from typing import List, Any
from tensorflow.python.keras.callbacks import EarlyStopping

from src.models import MLP
from experimental.datasets.managers import TestManager, Fold
from experimental.models.managers.model_manager import ModelManager


# noinspection PyMissingOrEmptyDocstring
class MLPManager(ModelManager):
    def __init__(self,
                 test_manager: TestManager,
                 epochs: int = 1000,
                 validation_split: float = 0.2,
                 early_stop: EarlyStopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
                 verbose: bool = False,
                 **kwargs):
        super(MLPManager, self).__init__(test_manager=test_manager,
                                         epochs=epochs,
                                         validation_split=validation_split,
                                         callbacks=[early_stop],
                                         verbose=verbose,
                                         **kwargs)

    def get_folds(self, num_folds: int, extrapolation: bool) -> List[Fold]:
        folds: List[Fold] = []
        for data, scalers in self.test_manager.dataset.load_data(num_folds=num_folds, extrapolation=extrapolation):
            folds.append(Fold(
                x=data['train'][0],
                y=data['train'][1],
                scalers=scalers,
                monotonicities=[],
                validation=data
            ))
        return folds

    def fit(self, fold: Fold) -> Any:
        model = MLP(output_act=self.output_act, h_units=self.h_units, scalers=fold.scalers)
        model.compile(loss=self.loss, optimizer=self.optimizer)
        model.fit(x=fold.x, y=fold.y, **self.fit_info)
        return model
