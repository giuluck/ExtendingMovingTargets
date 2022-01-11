import importlib.resources
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from moving_targets import MACS
from moving_targets.learners import LinearRegression, LogisticRegression
from moving_targets.metrics import DIDI, CrossEntropy, Accuracy, MSE, R2, Metric
from sklearn.preprocessing import OneHotEncoder

from src.managers.abstract_manager import AbstractManager, Config
from src.util.masters import FairRegression, FairClassification
from src.util.preprocessing import get_top_features, split_dataset, Scaler


class AdultManager(AbstractManager):
    """Data Manager for the adult dataset."""

    _IGNORED: List[str] = ['education', 'native-country']
    """List of ignored features."""

    @classmethod
    def data(cls) -> Dict[str, pd.DataFrame]:
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
    def model(cls, config: Config) -> MACS:
        return MACS(
            learner=LogisticRegression(max_iter=10000, **config.learner_kwargs),
            master=FairClassification(protected='race', **config.master_kwargs),
            metrics=cls.metrics(),
            stats=True,
            **config.macs_kwargs
        )

    @classmethod
    def metrics(cls) -> List[Metric]:
        didi = DIDI(classification=True, protected='race', percentage=True, name='constraint')
        return [CrossEntropy(name='loss'), Accuracy(name='metric'), didi]

    def __init__(self, config: Config):
        """
        :param config:
            A configuration instance to set the experiment parameters.
        """
        super(AdultManager, self).__init__(label='income',
                                           stratify=True,
                                           x_scaling='std',
                                           y_scaling=None,
                                           config=config)

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

    _MAX_NAN: float = 0.05
    """Relative maximal number of nan values for a column to be kept."""

    _N_FEATURES: int = 15
    """Number of input features to keep along with the protected one."""

    @classmethod
    def data(cls) -> Dict[str, pd.DataFrame]:
        # load dataset
        with importlib.resources.path('res', 'communities.csv') as filepath:
            df = pd.read_csv(filepath, header=0, sep=';', na_values='?', skipinitialspace=True)
        # drop rows with no associated attribute to be predicted, then build a set of ignored features which are either
        # in the default set or have too many missing values, and finally remove rows with missing values and cast
        df = df.dropna(subset=['violentPerPop'], axis=0).reset_index(drop=True)
        max_nan = int(cls._MAX_NAN * len(df))
        ignored = set(cls._IGNORED)
        for c in df.columns:
            num_nan = np.sum(df[c].isna())
            if num_nan > max_nan and c not in ['race', 'violentPerPop']:
                ignored.add(c)
        df = df.drop(list(ignored), axis=1).dropna(axis=0).reset_index(drop=True).astype(float)
        # perform feature selection on dataframe without protected feature (which is added at then end)
        x, y = df.drop(columns=['race', 'violentPerPop']), df[['violentPerPop']]
        ft = get_top_features(x=x, y=y, n=cls._N_FEATURES) + ['race', 'violentPerPop']
        return split_dataset(df[ft], test_size=0.2, val_size=0.0)

    @classmethod
    def model(cls, config: Config) -> MACS:
        return MACS(
            learner=LinearRegression(**config.learner_kwargs),
            master=FairRegression(protected='race', **config.master_kwargs),
            metrics=cls.metrics(),
            stats=True,
            **config.macs_kwargs
        )

    @classmethod
    def metrics(cls) -> List[Metric]:
        didi = DIDI(classification=False, protected='race', percentage=True, name='constraint')
        return [MSE(name='loss'), R2(name='metric'), didi]

    def __init__(self, config: Config):
        """
        :param config:
            A configuration instance to set the experiment parameters.
        """
        super(CommunitiesManager, self).__init__(label='violentPerPop',
                                                 stratify=True,
                                                 x_scaling='std',
                                                 y_scaling=None,
                                                 config=config)

    def get_scalers(self) -> Tuple[Scaler, Scaler]:
        # the x scaler uses the given default method for each feature but the protected one
        # (which is a categorical binary feature, thus it needs not scaling)
        x_scaler = Scaler() if self.x_scaling is None else Scaler(default_method=self.x_scaling, race=None)
        y_scaler = Scaler() if self.y_scaling is None else Scaler(default_method=self.y_scaling)
        return x_scaler, y_scaler
