"""Model Handler's Factory."""

from typing import Optional, List, Dict

from tensorflow.python.keras.callbacks import Callback

from experimental.utils.handlers.mlp_handler import MLPHandler
from experimental.utils.handlers.mt_handler import MTHandler
from experimental.utils.handlers.sbr_handler import SBRHandler, UnivariateSBRHandler
from experimental.utils.handlers.tfl_handler import TFLHandler
from moving_targets.metrics import Metric, MSE, CrossEntropy, Accuracy, R2
from src.datasets import AbstractManager
from src.util.typing import Augmented


# noinspection PyMissingOrEmptyDocstring
class HandlersFactory:
    def __init__(self,
                 manager: AbstractManager,
                 dataset: Optional[str] = None,
                 wandb_project: Optional[str] = 'moving_targets',
                 wandb_entity: Optional[str] = 'giuluck',
                 seed: int = 0,
                 loss: Optional[str] = None,
                 optimizer: str = 'adam',
                 output_act: Optional[str] = None,
                 h_units: Optional[List[int]] = None,
                 epochs: int = 1000,
                 batch_size: int = 32,
                 validation_split: float = 0.2,
                 callbacks: Optional[List[Callback]] = None,
                 verbose: bool = False,
                 num_augmented: Optional[Augmented] = None,
                 num_random: int = 0,
                 num_ground: Optional[int] = None,
                 monotonicities: str = 'group',
                 errors: str = 'raise',
                 master_kind: Optional[str] = None,
                 mt_metrics: Optional[List[Metric]] = None):
        self.manager: AbstractManager = manager
        self.dataset: Optional[str] = dataset
        self.wandb_project: Optional[str] = wandb_project
        self.wandb_entity: Optional[str] = wandb_entity
        self.seed: int = seed
        self.loss: Optional[str] = loss
        self.optimizer: str = optimizer
        self.output_act: Optional[str] = output_act
        self.h_units: Optional[List[int]] = h_units
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.callbacks: Optional[List[Callback]] = callbacks
        self.validation_split: float = validation_split
        self.verbose: bool = verbose
        self.num_augmented: Optional[Augmented] = num_augmented
        self.num_random: int = num_random
        self.num_ground: Optional[int] = num_ground
        self.monotonicities: str = monotonicities
        self.errors: str = errors
        self.master_kind: Optional[str] = master_kind
        self.mt_metrics: Optional[List[Metric]] = mt_metrics

    def _get_args(self, kwargs_dict: Dict, **kwargs_default) -> Dict:
        kwargs_default.update({
            'manager': self.manager,
            'dataset': self.dataset,
            'wandb_project': self.wandb_project,
            'wandb_entity': self.wandb_entity,
            'seed': self.seed
        })
        kwargs_default.update(kwargs_dict)
        return kwargs_default

    def get_mlp(self, wandb_name: Optional[str] = None, model: str = None, **kwargs) -> MLPHandler:
        return MLPHandler(**self._get_args(
            kwargs_dict=kwargs,
            wandb_name=wandb_name,
            model=model,
            loss=self.loss,
            optimizer=self.optimizer,
            output_act=self.output_act,
            h_units=self.h_units,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=self.callbacks,
            validation_split=self.validation_split,
            verbose=self.verbose
        ))

    def get_sbr(self, wandb_name: Optional[str] = None, model: str = None, **kwargs) -> SBRHandler:
        return SBRHandler(**self._get_args(
            kwargs_dict=kwargs,
            wandb_name=wandb_name,
            model=model,
            loss=self.loss,
            optimizer=self.optimizer,
            output_act=self.output_act,
            h_units=self.h_units,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=self.callbacks,
            validation_split=self.validation_split,
            num_augmented=self.num_augmented,
            num_random=self.num_random,
            num_ground=self.num_ground,
            verbose=self.verbose
        ))

    def get_univariate_sbr(self, wandb_name: Optional[str] = None, model: str = None, **kwargs) -> UnivariateSBRHandler:
        directions = list(self.manager.directions.values())
        assert len(directions) == 1
        return UnivariateSBRHandler(**self._get_args(
            kwargs_dict=kwargs,
            wandb_name=wandb_name,
            model=model,
            direction=directions[0],
            loss=self.loss,
            optimizer=self.optimizer,
            output_act=self.output_act,
            h_units=self.h_units,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=self.callbacks,
            validation_split=self.validation_split,
            verbose=self.verbose
        ))

    def get_tfl(self, wandb_name: Optional[str] = None, model: str = None, **kwargs) -> TFLHandler:
        return TFLHandler(**self._get_args(
            kwargs_dict=kwargs,
            wandb_name=wandb_name,
            model=model,
            optimizer=self.optimizer.capitalize(),
            epochs=self.epochs,
            batch_size=self.batch_size
        ))

    def get_mt(self, wandb_name: Optional[str] = None, model: str = None, **kwargs) -> MTHandler:
        return MTHandler(**self._get_args(
            kwargs_dict=kwargs,
            wandb_name=wandb_name,
            model=model,
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


# noinspection PyMissingOrEmptyDocstring
class RegressionFactory(HandlersFactory):
    def __init__(self, manager: AbstractManager, **kwargs):
        super(RegressionFactory, self).__init__(
            manager=manager,
            master_kind='regression',
            mt_metrics=[MSE(name='loss'), R2(name='metric')],
            loss='mse',
            output_act=None,
            **kwargs
        )


# noinspection PyMissingOrEmptyDocstring
class ClassificationFactory(HandlersFactory):
    def __init__(self, manager: AbstractManager, mt_evaluation_metric: Optional[Metric] = Accuracy(), **kwargs):
        super(ClassificationFactory, self).__init__(
            manager=manager,
            master_kind='classification',
            mt_metrics=[CrossEntropy(name='loss'), mt_evaluation_metric],
            loss='binary_crossentropy',
            output_act='sigmoid',
            **kwargs
        )
