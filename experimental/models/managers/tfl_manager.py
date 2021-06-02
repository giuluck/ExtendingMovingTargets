"""Tensorflow Lattice Manager."""
from typing import List, Any, Dict

import tensorflow_lattice as tfl

from experimental.datasets.managers import TestManager, Fold, RestaurantsTest, CarsTest, CarsUnivariateTest, \
    SyntheticTest, PuzzlesTest, DefaultTest, LawTest
from experimental.models.managers.model_manager import ModelManager
from src.models import TFL, ColumnInfo
from src.util.preprocessing import Scaler


# noinspection PyMissingOrEmptyDocstring
class TFLManager(ModelManager):
    DATASETS_INFO: Dict = {
        CarsTest: ('numeric', 'regression'),
        CarsUnivariateTest: ('numeric', 'regression'),
        SyntheticTest: ('numeric', 'regression'),
        PuzzlesTest: ('numeric', 'regression'),
        DefaultTest: ('numeric', 'binary'),
        LawTest: ('numeric', 'binary'),
        RestaurantsTest: ('categorical_restaurants', 'binary')
    }

    @staticmethod
    def get_numeric_features(columns: List[str], manager: TestManager) -> List[ColumnInfo]:
        return [
            ColumnInfo(
                name=column,
                lattice_size=2,
                monotonicity=manager.dataset.directions[column],
                pwl_calibration_num_keypoints=20,
                regularizer_configs=[tfl.configs.RegularizerConfig(name="calib_wrinkle", l2=1.0)]
            ) for column in columns
        ]

    @staticmethod
    def get_model_from_manager(fold: Fold, manager: TestManager, **kwargs) -> TFL:
        # RETRIEVE INFO BASED ON DATASET
        config, head = TFLManager.DATASETS_INFO[manager.__class__]
        # GET SCALERS
        if fold.scalers is None:
            x_scaler, y_scaler = None, None
        elif isinstance(fold.scalers, tuple):
            x_scaler, y_scaler = fold.scalers
        else:
            x_scaler, y_scaler = fold.scalers, None
        x_scaler = Scaler.get_default(num_features=len(fold.x.columns)) if x_scaler is None else x_scaler
        y_scaler = Scaler.get_default(num_features=1) if y_scaler is None else y_scaler
        # HANDLE FEATURE CONFIGURATIONS
        if config == 'categorical_restaurants':
            def pre_processing(x, y):
                x, y = x_scaler.transform(x), y_scaler.transform(y)
                drs = x[['D', 'DD', 'DDD', 'DDDD']].values.argmax(axis=1) + 1
                x['dollar_rating'] = ['D' * dr for dr in drs]
                x = x[['num_reviews', 'avg_rating', 'dollar_rating']]
                return x, y

            columns = TFLManager.get_numeric_features(columns=['num_reviews', 'avg_rating'], manager=manager) + [
                ColumnInfo(
                    kind='categorical',
                    name='dollar_rating',
                    lattice_size=2,
                    pwl_calibration_num_keypoints=4,
                    monotonicity=[('D', 'DD'), ('DDDD', 'D')]
                )
            ]
            post_processing = lambda y: y_scaler.inverse_transform(y)
        elif config == 'numeric':
            columns = TFLManager.get_numeric_features(columns=list(fold.x.columns), manager=manager)
            pre_processing = lambda x, y: (x_scaler.transform(x), y_scaler.transform(y))
            post_processing = lambda y: y_scaler.inverse_transform(y)
        else:
            raise ValueError(f"{config} is not a supported configuration.")
        # CREATE AND BUILD MODEL
        x_pre, y_pre = pre_processing(x=fold.x, y=fold.y)
        model = TFL(head=head, columns=columns, pre_processing=pre_processing, post_processing=post_processing)
        model.build(x=x_pre, y=y_pre, seed=manager.seed, **kwargs)
        return model

    def __init__(self,
                 test_manager: TestManager,
                 epochs: int = 1000,
                 batch_size: int = 32,
                 **kwargs):
        super(TFLManager, self).__init__(test_manager=test_manager,
                                         epochs=epochs,
                                         batch_size=batch_size,
                                         **kwargs)

    def get_folds(self, num_folds: int, extrapolation: bool, compute_monotonicities: bool = False) -> List[Fold]:
        return [Fold(
            x=data['train'][0],
            y=data['train'][1],
            scalers=scalers,
            monotonicities=[],
            validation=data
        ) for data, scalers in self.test_manager.dataset.load_data(num_folds=num_folds, extrapolation=extrapolation)]

    def fit(self, fold: Fold) -> Any:
        model = TFLManager.get_model_from_manager(fold, self.test_manager, optimizer=self.optimizer.capitalize())
        model.fit(x=fold.x, y=fold.y, **self.fit_info)
        return model
