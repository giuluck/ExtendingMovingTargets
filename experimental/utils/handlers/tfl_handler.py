"""Tensorflow Lattice Handler."""
from typing import List, Any, Dict

import tensorflow_lattice as tfl

from experimental.utils.handlers import AbstractHandler, Fold
from src.datasets import AbstractManager, CarsManager, DefaultManager, LawManager, PuzzlesManager, RestaurantsManager, \
    SyntheticManager
from src.models import TFL, ColumnInfo
from src.util.preprocessing import Scaler


# noinspection PyMissingOrEmptyDocstring
class TFLHandler(AbstractHandler):
    DATASETS_INFO: Dict = {
        CarsManager: ('numeric', 'regression'),
        SyntheticManager: ('numeric', 'regression'),
        PuzzlesManager: ('numeric', 'regression'),
        DefaultManager: ('numeric', 'binary'),
        LawManager: ('numeric', 'binary'),
        RestaurantsManager: ('categorical_restaurants', 'binary')
    }

    @staticmethod
    def get_features(columns: List[str], manager: AbstractManager) -> List[ColumnInfo]:
        expected = list(manager.directions.keys())
        assert set(expected) <= set(columns), f"some of the expected columns {expected} are not present in {columns}"
        # handle categorical features (formatted as "<category_name>: <category_value>")
        columns_info = {}
        for c in columns:
            if ': ' in c:
                k, v = c.split(': ')
                columns_info[k] = 1 + (columns_info.get(k) or 0)
            else:
                columns_info[c] = 0
        # create numeric or categorical columns based on the kind of inputs
        tfl_columns = []
        for column, length in columns_info.items():
            if length == 0:
                info = ColumnInfo(
                    name=column,
                    lattice_size=2,
                    monotonicity=manager.directions.get(column) or 0,
                    pwl_calibration_num_keypoints=25,
                    regularizer_configs=[tfl.configs.RegularizerConfig(name="calib_wrinkle", l2=1.0)]
                )
            else:
                info = ColumnInfo(
                    kind='categorical',
                    name=column,
                    lattice_size=2,
                    pwl_calibration_num_keypoints=length,
                )
            tfl_columns.append(info)
        return tfl_columns

    def __init__(self,
                 manager: AbstractManager,
                 optimizer: str = 'Adam',
                 epochs: int = 1000,
                 batch_size: int = 32,
                 **kwargs):
        super(TFLHandler, self).__init__(manager=manager, **kwargs)
        self.optimizer: str = optimizer
        self.epochs: int = epochs
        self.batch_size: int = batch_size

    def fit(self, fold: Fold) -> Any:
        # RETRIEVE INFO BASED ON DATASET
        config, head = TFLHandler.DATASETS_INFO[self.manager.__class__]
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
                x, y = x_scaler.transform(x), None if y is None else y_scaler.transform(y)
                drs = x[['D', 'DD', 'DDD', 'DDDD']].values.argmax(axis=1) + 1
                x['dollar_rating'] = ['D' * dr for dr in drs]
                x = x[['num_reviews', 'avg_rating', 'dollar_rating']]
                return x, y

            columns = TFLHandler.get_features(columns=['num_reviews', 'avg_rating'], manager=self.manager) + [
                ColumnInfo(
                    kind='categorical',
                    name='dollar_rating',
                    lattice_size=2,
                    pwl_calibration_num_keypoints=4,
                    monotonicity=[('D', 'DD'), ('DDDD', 'D')]
                )
            ]
        elif config == 'numeric':
            def pre_processing(x, y):
                x, y = x_scaler.transform(x), None if y is None else y_scaler.transform(y)
                # retrieve categorical features (formatted as "<category_name>: <category_value>")
                categories = {}
                for c in x.columns:
                    if ': ' in c:
                        k, v = c.split(': ')
                        categories[k] = (categories.get(k) or []) + [v]
                # get category value, map it to the respective value, then drop the one-hot encoded columns
                for category, values in categories.items():
                    cols = [f'{category}: {value}' for value in values]
                    x[category] = x[cols].values.argmax(axis=1)
                    x[category] = x[category].map({i: v for i, v in enumerate(values)})
                    x = x.drop(columns=cols)
                return x, y

            columns = TFLHandler.get_features(columns=list(fold.x.columns), manager=self.manager)
        else:
            raise ValueError(f"{config} is not a supported configuration.")
        # CREATE AND BUILD MODEL
        x_pre, y_pre = pre_processing(x=fold.x, y=fold.y)
        model = TFL(head=head, columns=columns, pre_processing=pre_processing,
                    post_processing=lambda y: y_scaler.inverse_transform(y))
        model.build(x=x_pre, y=y_pre, optimizer=self.optimizer, seed=self.seed)
        model.fit(x=fold.x, y=fold.y, epochs=self.epochs, batch_size=self.batch_size)
        return model
