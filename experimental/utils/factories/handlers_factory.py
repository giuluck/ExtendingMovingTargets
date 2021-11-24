"""Handlers Factory."""

from typing import Optional, List, Callable, Union, Dict

from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

from experimental.utils.handlers import default_config
from experimental.utils.handlers.mlp_handler import MLPHandler
from experimental.utils.handlers.mt_handler import MTHandler
from experimental.utils.handlers.sbr_handler import SBRHandler, UnivariateSBRHandler
from experimental.utils.handlers.tfl_handler import TFLHandler
from moving_targets.metrics import Metric
from src.datasets import AbstractManager
from src.util.typing import Augmented


class HandlersFactory:
    """Factory class that returns model handlers."""

    def __init__(self,
                 manager: AbstractManager,
                 dataset: Optional[str] = None,
                 wandb_project: Optional[str] = 'moving_targets',
                 wandb_entity: Optional[str] = 'giuluck',
                 wandb_config: Optional[Callable] = default_config,
                 seed: int = 0,
                 loss: Optional[str] = None,
                 optimizer: str = 'adam',
                 run_eagerly: bool = False,
                 output_act: Optional[str] = None,
                 h_units: Optional[List[int]] = None,
                 epochs: int = 1000,
                 batch_size: int = 32,
                 validation_split: float = 0.2,
                 keras_callbacks: Optional[List[Callback]] = None,
                 verbose: bool = False,
                 alpha: Union[None, float, OptimizerV2] = None,
                 regularizer_act: Optional[Callable] = None,
                 num_augmented: Optional[Augmented] = None,
                 num_random: int = 0,
                 num_ground: Optional[int] = None,
                 monotonicities: str = 'group',
                 errors: str = 'raise',
                 master_kind: Optional[str] = None,
                 mt_metrics: Optional[List[Metric]] = None):
        """
        :param manager:
            The dataset manager.

        :param dataset:
            The dataset name.

        :param wandb_project:
            The Weights&Biases project name.

        :param wandb_entity:
            The Weights&Biases project entity.

        :param wandb_config:
            The Weights&Biases configuration function which is in charge of returning the configuration dictionary.

        :param seed:
            The random seed.

        :param loss:
            The neural network/learner loss (ignored for TFL).

        :param optimizer:
            The neural network/learner/tfl optimizer.

        :param run_eagerly:
            Whether or not to run tensorflow in eager mode (ignored for TFL).

        :param output_act:
            The neural network/learner output activation (ignored for TFL).

        :param h_units:
            The neural network/learner list of hidden units (ignored for TFL).

        :param epochs:
            The neural network/learner/tfl training epochs.

        :param batch_size:
            The neural network/learner/tfl batch size.

        :param validation_split:
            The neural network validation split (ignored for TFL and MT).

        :param keras_callbacks:
            The list of keras callbacks for the neural network (ignored for TFL and MT).

        :param verbose:
            The neural network/moving targets verbosity (ignored for TFL).

        :param alpha:
            The alpha value for balancing compiled and regularized loss (ignored for MLP, TFL and MT).

        :param regularizer_act:
            The regularizer activation function (ignored for MLP, TFL and MT).

        :param num_augmented:
            The number of augmented samples (ignored for MLP and TFL).

        :param num_random:
            The number of unlabelled random samples added to the original dataset (ignored for MLP and TFL).

        :param num_ground:
            The number of samples taken from the original dataset (ignored for MLP and TFL).

        :param monotonicities:
            The monotonicity computation modality (ignored for MLP, SBR, and TFL).

        :param errors:
            Error strategy when dropping columns due to monotonicity computation (ignored for MLP, SBR, and TFL).

        :param master_kind:
            The Moving Targets' master kind, either 'regression' or 'classification' (ignored for MLP, SBR, and TFL).

        :param mt_metrics:
            A list of `Metric` instances to evaluate the final MT solution (ignored for MLP, SBR, and TFL).
        """

        self.manager: AbstractManager = manager
        """The dataset manager."""

        self.dataset: Optional[str] = dataset
        """The dataset name."""

        self.wandb_project: Optional[str] = wandb_project
        """The Weights&Biases project name."""

        self.wandb_entity: Optional[str] = wandb_entity
        """The Weights&Biases project entity."""

        self.wandb_config: Optional[Callable] = wandb_config
        """The Weights&Biases configuration function which is in charge of returning the configuration dictionary."""

        self.seed: int = seed
        """The random seed."""

        self.loss: Optional[str] = loss
        """The neural network/learner loss (ignored for TFL)."""

        self.optimizer: str = optimizer
        """The neural network/learner/tfl optimizer."""

        self.run_eagerly: bool = run_eagerly
        """Whether or not to run tensorflow in eager mode."""

        self.output_act: Optional[str] = output_act
        """The neural network/learner output activation (ignored for TFL)."""

        self.h_units: Optional[List[int]] = h_units
        """The neural network/learner list of hidden units (ignored for TFL)."""

        self.epochs: int = epochs
        """The neural network/learner/tfl training epochs."""

        self.batch_size: int = batch_size
        """The neural network/learner/tfl batch size."""

        self.keras_callbacks: Optional[List[Callback]] = keras_callbacks
        """The neural network validation split (ignored for TFL and MT)."""

        self.validation_split: float = validation_split
        """The list of keras callbacks for the neural network (ignored for TFL and MT)."""

        self.verbose: bool = verbose
        """The neural network/moving targets verbosity (ignored for TFL)."""

        self.alpha: Union[None, float, OptimizerV2] = alpha
        """The alpha value for balancing compiled and regularized loss."""

        self.regularizer_act: Optional[Callable] = regularizer_act
        """The regularizer activation function."""

        self.num_augmented: Optional[Augmented] = num_augmented
        """The number of augmented samples (ignored for MLP and TFL)."""

        self.num_random: int = num_random
        """The number of unlabelled random samples added to the original dataset (ignored for MLP and TFL)."""

        self.num_ground: Optional[int] = num_ground
        """The number of samples taken from the original dataset (ignored for MLP and TFL)."""

        self.monotonicities: str = monotonicities
        """The monotonicity computation modality (ignored for MLP, SBR, and TFL)."""

        self.errors: str = errors
        """Error strategy when dropping columns due to monotonicity computation (ignored for MLP, SBR, and TFL)."""

        self.master_kind: Optional[str] = master_kind
        """The Moving Targets' master kind, either 'regression' or 'classification' (ignored for MLP, SBR, and TFL)."""

        self.mt_metrics: Optional[List[Metric]] = mt_metrics
        """A list of `Metric` instances to evaluate the final MT solution (ignored for MLP, SBR, and TFL)."""

    @staticmethod
    def _get_args(additional_kwargs: Dict, **default_args) -> Dict:
        """Replace passed additional arguments in the dictionary of default arguments.

        :param additional_kwargs:
            The dictionary of additional arguments.

        :param default_args:
            The named default arguments.

        :return:
            A merged dictionary where additional arguments replace named default arguments.
        """
        default_args.update(additional_kwargs)
        return default_args

    def get_mlp(self, wandb_name: Optional[str] = None, model: str = None, **additional_kwargs) -> MLPHandler:
        """Builds an MLP model handler.

        :param wandb_name:
            The Weights&Biases run name.

        :param model:
            The name of the model.

        :param additional_kwargs:
            Any argument that substitutes a default one.

        :return:
            An `MLPHandler` object.
        """
        return MLPHandler(**HandlersFactory._get_args(
            additional_kwargs=additional_kwargs,
            manager=self.manager,
            loss=self.loss,
            model=model,
            dataset=self.dataset,
            wandb_name=wandb_name,
            wandb_project=self.wandb_project,
            wandb_entity=self.wandb_entity,
            wandb_config=self.wandb_config,
            seed=self.seed,
            optimizer=self.optimizer,
            run_eagerly=self.run_eagerly,
            output_act=self.output_act,
            h_units=self.h_units,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=self.keras_callbacks,
            validation_split=self.validation_split,
            verbose=self.verbose,
        ))

    def get_sbr(self, wandb_name: Optional[str] = None, model: str = None, **additional_kwargs) -> SBRHandler:
        """Builds an SBR model handler.

        :param wandb_name:
            The Weights&Biases run name.

        :param model:
            The name of the model.

        :param additional_kwargs:
            Any argument that substitutes a default one.

        :return:
            An `SBRHandler` object.
        """
        return SBRHandler(**HandlersFactory._get_args(
            additional_kwargs=additional_kwargs,
            manager=self.manager,
            loss=self.loss,
            model=model,
            dataset=self.dataset,
            wandb_name=wandb_name,
            wandb_project=self.wandb_project,
            wandb_entity=self.wandb_entity,
            wandb_config=self.wandb_config,
            seed=self.seed,
            optimizer=self.optimizer,
            run_eagerly=self.run_eagerly,
            output_act=self.output_act,
            h_units=self.h_units,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=self.keras_callbacks,
            verbose=self.verbose,
            alpha=self.alpha,
            regularizer_act=self.regularizer_act,
            num_augmented=self.num_augmented,
            num_random=self.num_random,
            num_ground=self.num_ground
        ))

    def get_univariate_sbr(self, wandb_name: Optional[str] = None, model: str = None,
                           **additional_kwargs) -> UnivariateSBRHandler:
        """Builds a Univariate SBR model handler.

        :param wandb_name:
            The Weights&Biases run name.

        :param model:
            The name of the model.

        :param additional_kwargs:
            Any argument that substitutes a default one.

        :return:
            An `UnivariateSBRHandler` object.
        """
        directions = list(self.manager.directions.values())
        assert len(directions) == 1
        return UnivariateSBRHandler(**HandlersFactory._get_args(
            additional_kwargs=additional_kwargs,
            manager=self.manager,
            loss=self.loss,
            direction=directions[0],
            model=model,
            dataset=self.dataset,
            wandb_name=wandb_name,
            wandb_project=self.wandb_project,
            wandb_entity=self.wandb_entity,
            wandb_config=self.wandb_config,
            seed=self.seed,
            optimizer=self.optimizer,
            run_eagerly=self.run_eagerly,
            output_act=self.output_act,
            h_units=self.h_units,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=self.keras_callbacks,
            validation_split=self.validation_split,
            verbose=self.verbose,
            alpha=self.alpha,
            regularizer_act=self.regularizer_act
        ))

    def get_tfl(self, wandb_name: Optional[str] = None, model: str = None, **additional_kwargs) -> TFLHandler:
        """Builds an TFL model handler.

        :param wandb_name:
            The Weights&Biases run name.

        :param model:
            The name of the model.

        :param additional_kwargs:
            Any argument that substitutes a default one.

        :return:
            An `TFLHandler` object.
        """
        return TFLHandler(**HandlersFactory._get_args(
            additional_kwargs=additional_kwargs,
            manager=self.manager,
            model=model,
            dataset=self.dataset,
            wandb_name=wandb_name,
            wandb_project=self.wandb_project,
            wandb_entity=self.wandb_entity,
            wandb_config=self.wandb_config,
            seed=self.seed,
            optimizer=self.optimizer.capitalize(),
            epochs=self.epochs,
            batch_size=self.batch_size
        ))

    def get_mt(self, wandb_name: Optional[str] = None, model: str = None, **additional_kwargs) -> MTHandler:
        """Builds an MT model handler.

        :param wandb_name:
            The Weights&Biases run name.

        :param model:
            The name of the model.

        :param additional_kwargs:
            Any additional custom argument to be passed to `MTHandler` or that substitutes a default one.

        :return:
            An `MTHandler` object.
        """
        return MTHandler(**HandlersFactory._get_args(
            additional_kwargs=additional_kwargs,
            manager=self.manager,
            model=model,
            dataset=self.dataset,
            wandb_name=wandb_name,
            wandb_project=self.wandb_project,
            wandb_entity=self.wandb_entity,
            wandb_config=self.wandb_config,
            seed=self.seed,
            aug_num_augmented=self.num_augmented,
            aug_num_random=self.num_random,
            aug_num_ground=self.num_ground,
            mnt_kind=self.monotonicities,
            mnt_errors=self.errors,
            mt_metrics=self.mt_metrics,
            mt_verbose=self.verbose,
            lrn_loss=self.loss,
            lrn_optimizer=self.optimizer,
            lrn_output_act=self.output_act,
            lrn_h_units=self.h_units,
            lrn_batch_size=self.batch_size,
            mst_master_kind=self.master_kind
        ))
