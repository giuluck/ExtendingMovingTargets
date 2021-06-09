"""Multi-Layer Perceptron Handler."""

from typing import List, Any, Optional

from tensorflow.python.keras.callbacks import EarlyStopping, Callback

from experimental.utils.handlers import AbstractHandler, Fold
from src.datasets import AbstractManager
from src.models import MLP


# noinspection PyMissingOrEmptyDocstring
class MLPHandler(AbstractHandler):
    def __init__(self,
                 manager: AbstractManager,
                 loss: str,
                 optimizer: str = 'adam',
                 run_eagerly: bool = False,
                 output_act: Optional[str] = None,
                 h_units: Optional[List[int]] = None,
                 epochs: int = 1000,
                 batch_size: int = 32,
                 validation_split: float = 0.2,
                 callbacks: Optional[List[Callback]] = None,
                 verbose: bool = False,
                 **kwargs):
        super(MLPHandler, self).__init__(manager=manager, **kwargs)
        self.loss: str = loss
        self.optimizer: str = optimizer
        self.run_eagerly: bool = run_eagerly
        self.output_act: Optional[str] = output_act
        self.h_units: Optional[List[int]] = h_units
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.validation_split: float = validation_split
        if callbacks is None:
            self.callbacks: List[Callback] = [EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)]
        else:
            self.callbacks: List[Callback] = callbacks
        self.verbose: bool = verbose

    def fit(self, fold: Fold) -> Any:
        model = MLP(output_act=self.output_act, h_units=self.h_units, scalers=fold.scalers)
        model.compile(loss=self.loss, optimizer=self.optimizer, run_eagerly=self.run_eagerly)
        model.fit(x=fold.x, y=fold.y, validation_split=self.validation_split, epochs=self.epochs,
                  batch_size=self.batch_size, callbacks=self.callbacks, verbose=self.verbose)
        return model
