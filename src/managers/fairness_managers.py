import importlib.resources
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from moving_targets import MACS
from moving_targets.learners import LinearRegression, LogisticRegression
from moving_targets.metrics import DIDI, CrossEntropy, Accuracy, MSE, R2, Metric
from sklearn.preprocessing import OneHotEncoder

from src.managers.abstract_manager import AbstractManager
from src.util.masters import FairRegression, FairClassification
from src.util.preprocessing import get_top_features, split_dataset, Scaler


class AdultManager(AbstractManager):
    """Data Manager for the adult dataset."""

    _IGNORED: List[str] = ['education', 'native-country']
    """List of ignored features."""

    @classmethod
    def data(cls, **data_kwargs) -> Dict[str, pd.DataFrame]:
        with importlib.resources.path('res', 'adult.csv') as filepath:
            df = pd.read_csv(filepath, header=0, sep=';', na_values='?', skipinitialspace=True)
        df = df.dropna(axis=0).reset_index(drop=True)
        # manage target and protected features
        target = pd.Series(df['income'].astype('category').cat.codes, name='income')
        protected = pd.get_dummies(df['race'], prefix='race')
        # manage other input features
        features = df.drop(columns=cls._IGNORED + ['race', 'income'])
        encoder = OneHotEncoder(drop=None, dtype=np.float, sparse=False)
        categorical_features = encoder.fit_transform(features.select_dtypes(include=['object']))
        categorical_features = pd.DataFrame(categorical_features, columns=encoder.get_feature_names_out())
        numerical_features = features.select_dtypes(exclude=['object'])
        # obtain output data by concatenating features
        df = pd.concat((numerical_features, categorical_features, protected), axis=1).astype(float)
        df['income'] = target.astype('category').cat.codes
        return split_dataset(df, stratify=df['income'], test_size=0.2, val_size=0.0)

    @classmethod
    def model(cls, **model_kwargs) -> MACS:
        return MACS(
            init_step=model_kwargs['init_step'],
            learner=LogisticRegression(max_iter=10000),
            master=FairClassification(
                protected='race',
                backend=model_kwargs['backend'],
                loss=model_kwargs['loss'],
                alpha=model_kwargs['alpha'],
                beta=model_kwargs['beta'],
                adaptive=model_kwargs['adaptive']
            ),
            metrics=cls.metrics(),
            stats=True
        )

    @classmethod
    def metrics(cls) -> List[Metric]:
        didi = DIDI(classification=True, protected='race', percentage=True, name='constraint')
        return [CrossEntropy(name='loss'), Accuracy(name='metric'), didi]

    def __init__(self, seed: int = 0, **kwargs):
        """
        :param seed:
            The random seed.

        :param kwargs:
            Any additional argument to be passed to the 'model()' function.
        """
        super(AdultManager, self).__init__(label='income',
                                           stratify=True,
                                           x_scaling='std',
                                           y_scaling=None,
                                           seed=seed,
                                           **kwargs)

    def get_scalers(self) -> Tuple[Scaler, Scaler]:
        # the x scaler uses the given default method for numeric features while the categorical ones (which have an
        # underscore in the name) are not scaled
        custom_methods = {c: None for c in self.train.columns if '_' in c}
        x_scaler = Scaler() if self.x_scaling is None else Scaler(default_method=self.x_scaling, **custom_methods)
        y_scaler = Scaler() if self.y_scaling is None else Scaler(default_method=self.y_scaling)
        return x_scaler, y_scaler


class CommunitiesManager(AbstractManager):
    """Data Manager for the communities dataset."""

    # noinspection SpellCheckingInspection
    _IGNORED: List[str] = ['fold', 'state', 'murders', 'murdPerPop', 'rapes', 'rapesPerPop', 'robberies', 'robbbPerPop',
                           'communityname', 'assaults', 'assaultPerPop', 'burglaries', 'burglPerPop', 'larcenies',
                           'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop', 'nonViolPerPop']
    """List of ignored features."""

    @classmethod
    def data(cls, **data_kwargs) -> Dict[str, pd.DataFrame]:
        # check data kwargs
        max_nan = data_kwargs.get('max_nan') or 0.05
        n_features = data_kwargs.get('n_features') or 15
        assert max_nan >= 0, "'max_nan' should be a positive value, either float or integer"
        assert n_features >= 0, "'n_features' should be a positive integer value"
        # load dataset
        with importlib.resources.path('res', 'communities.csv') as filepath:
            df = pd.read_csv(filepath, header=0, sep=';', na_values='?', skipinitialspace=True)
        max_nan = max_nan if isinstance(max_nan, int) else int(max_nan * len(df))
        # drop rows with no associated attribute to be predicted, then build a set of ignored features which are either
        # in the default set or have too many missing values, and finally remove rows with missing values and cast
        df = df.dropna(subset=['violentPerPop'], axis=0).reset_index(drop=True)
        ignored = set(cls._IGNORED)
        for c in df.columns:
            num_nan = np.sum(df[c].isna())
            if num_nan > max_nan and c not in ['race', 'violentPerPop']:
                ignored.add(c)
        df = df.drop(list(ignored), axis=1).dropna(axis=0).reset_index(drop=True).astype(float)
        # perform feature selection on dataframe without protected feature (which is added at then end)
        x, y = df.drop(columns=['race', 'violentPerPop']), df[['violentPerPop']]
        ft = get_top_features(x=x, y=y, n=n_features) + ['race', 'violentPerPop']
        return split_dataset(df[ft], test_size=0.2, val_size=0.0)

    @classmethod
    def model(cls, **model_kwargs) -> MACS:
        return MACS(
            init_step=model_kwargs['init_step'],
            learner=LinearRegression(),
            master=FairRegression(
                protected='race',
                backend=model_kwargs['backend'],
                loss=model_kwargs['loss'],
                alpha=model_kwargs['alpha'],
                beta=model_kwargs['beta'],
                adaptive=model_kwargs['adaptive']
            ),
            metrics=cls.metrics(),
            stats=True
        )

    @classmethod
    def metrics(cls) -> List[Metric]:
        didi = DIDI(classification=False, protected='race', percentage=True, name='constraint')
        return [MSE(name='loss'), R2(name='metric'), didi]

    def __init__(self, seed: int = 0, **kwargs):
        """
        :param seed:
            The random seed.

        :param kwargs:
            Any additional argument to be passed to the 'model()' function.
        """
        super(CommunitiesManager, self).__init__(label='violentPerPop',
                                                 stratify=True,
                                                 x_scaling='std',
                                                 y_scaling=None,
                                                 seed=seed,
                                                 **kwargs)

    def get_scalers(self) -> Tuple[Scaler, Scaler]:
        # the x scaler uses the given default method for each feature but the protected one
        # (which is a categorical binary feature, thus it needs not scaling)
        x_scaler = Scaler() if self.x_scaling is None else Scaler(default_method=self.x_scaling, race=None)
        y_scaler = Scaler() if self.y_scaling is None else Scaler(default_method=self.y_scaling)
        return x_scaler, y_scaler
