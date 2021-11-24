"""Semantic-Based Regularizer Handler."""

from typing import Any, Callable, Optional, Union, List

import numpy as np
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

from experimental.utils.handlers import default_config
from experimental.utils.handlers.mlp_handler import MLPHandler, Fold
from moving_targets.callbacks import Callback
from src.datasets import AbstractManager
from src.models import SBR, SBRBatchGenerator, UnivariateSBR
from src.util.typing import Augmented


class SBRHandler(MLPHandler):
    """Semantic-Based Regularizer Model Handler."""

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
                 verbose: bool = False,
                 alpha: Union[None, float, OptimizerV2] = None,
                 regularizer_act: Optional[Callable] = None,
                 num_augmented: Optional[Augmented] = None,
                 num_random: int = 0,
                 num_ground: Optional[int] = None):
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

        :param alpha:
            The alpha value for balancing compiled and regularized loss.

        :param regularizer_act:
            The regularizer activation function.

        :param num_augmented:
            The number of augmented samples.

        :param num_random:
            The number of unlabelled random samples added to the original dataset.

        :param num_ground:
            The number of samples taken from the original dataset (the remaining ones are ignored).

        """
        super(SBRHandler, self).__init__(manager=manager, loss=loss, model=model, dataset=dataset,
                                         wandb_name=wandb_name, wandb_project=wandb_project, wandb_entity=wandb_entity,
                                         wandb_config=wandb_config, seed=seed, optimizer=optimizer,
                                         run_eagerly=run_eagerly, output_act=output_act, h_units=h_units,
                                         epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                                         callbacks=callbacks, verbose=verbose)

        self.alpha: Union[None, float, OptimizerV2] = alpha
        """The alpha value for balancing compiled and regularized loss."""

        self.regularizer_act: Optional[Callable] = regularizer_act
        """The regularizer activation function."""

        self.num_augmented: Optional[Augmented] = num_augmented
        """The number of augmented samples."""

        self.num_random: int = num_random
        """The number of unlabelled random samples added to the original dataset."""

        self.num_ground: Optional[int] = num_ground
        """The number of samples taken from the original dataset (the remaining ones are ignored)."""

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


class UnivariateSBRHandler(MLPHandler):
    """Univariate Semantic-Based Regularizer Model Handler."""
    def __init__(self,
                 manager: AbstractManager,
                 loss: str,
                 direction: int,
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
                 verbose: bool = False,
                 alpha: Union[None, float, OptimizerV2] = None,
                 regularizer_act: Optional[Callable] = None):
        """
        :param manager:
            The dataset manager.

        :param loss:
            The neural network loss function.

        :param direction:
            The monotonicity direction.

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

        :param alpha:
            The alpha value for balancing compiled and regularized loss.

        :param regularizer_act:
            The regularizer activation function.

        """
        super(UnivariateSBRHandler, self).__init__(manager=manager, loss=loss, model=model, dataset=dataset,
                                                   wandb_name=wandb_name, wandb_project=wandb_project,
                                                   wandb_entity=wandb_entity, wandb_config=wandb_config, seed=seed,
                                                   optimizer=optimizer, run_eagerly=run_eagerly, output_act=output_act,
                                                   h_units=h_units, epochs=epochs, batch_size=batch_size,
                                                   validation_split=validation_split, callbacks=callbacks,
                                                   verbose=verbose)
        self.direction: int = direction
        """The monotonicity direction."""

        self.alpha: Union[None, float, OptimizerV2] = alpha
        """The alpha value for balancing compiled and regularized loss."""

        self.regularizer_act: Optional[Callable] = regularizer_act
        """The regularizer activation function."""

    def fit(self, fold: Fold) -> Any:
        model = UnivariateSBR(output_act=self.output_act, h_units=self.h_units, alpha=self.alpha, input_dim=1,
                              regularizer_act=self.regularizer_act, direction=self.direction, scalers=fold.scalers)
        model.compile(loss=self.loss, optimizer=self.optimizer, run_eagerly=self.run_eagerly)
        model.fit(x=fold.x, y=fold.y, validation_split=self.validation_split, epochs=self.epochs,
                  batch_size=self.batch_size, callbacks=self.callbacks, verbose=self.verbose)
        return model
