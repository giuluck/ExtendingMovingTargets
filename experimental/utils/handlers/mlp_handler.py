"""Multi-Layer Perceptron Handler."""

from typing import List, Any, Optional, Callable

from tensorflow.python.keras.callbacks import EarlyStopping, Callback

from experimental.utils.handlers import AbstractHandler, Fold, default_config
from src.datasets import AbstractManager
from src.models import MLP


class MLPHandler(AbstractHandler):
    """Multi-Layer Perceptron Model Handler."""

    def __init__(self,
                 manager: AbstractManager,
                 loss: str,
                 model: Optional[str] = None,
                 dataset: Optional[str] = None,
                 wandb_name: Optional[str] = None,
                 wandb_project: Optional[str] = 'moving_targets',
                 wandb_entity: Optional[str] = 'giuluck',
                 wandb_config: Callable = default_config,
                 seed: int = 0,
                 optimizer: str = 'adam',
                 run_eagerly: bool = False,
                 output_act: Optional[str] = None,
                 h_units: Optional[List[int]] = None,
                 epochs: int = 1000,
                 batch_size: int = 32,
                 validation_split: float = 0.2,
                 callbacks: Optional[List[Callback]] = None,
                 verbose: bool = False):
        """
        :param manager:
            The dataset manager.

        :param loss:
            The neural network loss function.

        :param model:
            The machine learning model name.

        :param dataset:
            The dataset name.

        :param wandb_name:
            The Weights&Biases run name. If None, no Weights&Biases instance is created.

        :param wandb_project:
            The Weights&Biases project name. If wandb_name is None, this is ignored.

        :param wandb_entity:
            The Weights&Biases entity name. If wandb_name is None, this is ignored.

        :param wandb_config:
            The Weights&Biases configuration function which is in charge of returning the configuration dictionary.

        :param seed:
            The random seed.

        :param optimizer:
            The neural network optimizer.

        :param run_eagerly:
            Whether or not to run tensorflow in eager mode.

        :param output_act:
            The neural network output activation.

        :param h_units:
            The list of neural network hidden units.

        :param epochs:
            The number of training epochs.

        :param batch_size:
            The batch size for neural network training.

        :param validation_split:
            The validation split for neural network training.

        :param callbacks:
            The list of keras callbacks for the `MLP` object. If None, the default `EarlyStopping` callback.

        :param verbose:
            Whether or not to print information during the neural network training.
        """
        super(MLPHandler, self).__init__(manager=manager, model=model, dataset=dataset, wandb_name=wandb_name,
                                         wandb_project=wandb_project, wandb_entity=wandb_entity,
                                         wandb_config=wandb_config, seed=seed)
        self.loss: str = loss
        """The MLP loss function."""

        self.optimizer: str = optimizer
        """The neural network optimizer."""

        self.run_eagerly: bool = run_eagerly
        """Whether or not to run tensorflow in eager mode."""

        self.output_act: Optional[str] = output_act
        """The neural network output activation."""

        self.h_units: Optional[List[int]] = h_units
        """The list of neural network hidden units."""

        self.epochs: int = epochs
        """The number of training epochs."""

        self.batch_size: int = batch_size
        """The batch size for neural network training."""

        self.validation_split: float = validation_split
        """The validation split for neural network training."""

        self.callbacks: List[Callback] = []
        """The list of keras callbacks for the `MLP` object. If None, the default `EarlyStopping` callback."""

        self.verbose: bool = verbose
        """Whether or not to print information during the neural network training."""

        if callbacks is None:
            self.callbacks = [EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)]
        else:
            self.callbacks = callbacks

    def fit(self, fold: Fold) -> Any:
        model = MLP(output_act=self.output_act, h_units=self.h_units, scalers=fold.scalers)
        model.compile(loss=self.loss, optimizer=self.optimizer, run_eagerly=self.run_eagerly)
        model.fit(x=fold.x, y=fold.y, validation_split=self.validation_split, epochs=self.epochs,
                  batch_size=self.batch_size, callbacks=self.callbacks, verbose=self.verbose)
        return model
