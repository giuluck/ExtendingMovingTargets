"""Cars Data Manager."""

from typing import Tuple, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

from src.datasets.abstract_manager import AbstractManager
from src.util.cleaning import FeatureInfo, clean_dataframe
from src.util.preprocessing import split_dataset
from src.util.typing import Augmented, SamplingFunctions, Method, Rng, Figsize, TightLayout


class CarsManager(AbstractManager):
    """Data Manager for the Cars Dataset."""

    FEATURES: Dict[str, FeatureInfo] = {
        'Sales in thousands': FeatureInfo(dtype='float', alias='sales'),
        'Price in thousands': FeatureInfo(dtype='float', alias='price'),
        'Engine size': FeatureInfo(dtype='float', alias='engine_size'),
        'Horsepower': FeatureInfo(dtype='float', alias='horsepower'),
        'Wheelbase': FeatureInfo(dtype='float', alias='wheelbase'),
        'Width': FeatureInfo(dtype='float', alias='width'),
        'Length': FeatureInfo(dtype='float', alias='length'),
        'Curb weight': FeatureInfo(dtype='float', alias='curb_weight'),
        'Fuel capacity': FeatureInfo(dtype='float', alias='fuel_capacity'),
        'Fuel efficiency': FeatureInfo(dtype='float', alias='fuel_efficiency'),
        'Manufacturer': FeatureInfo(dtype='category', alias='manufacturer'),
        'Vehicle type': FeatureInfo(dtype='category', alias='vehicle')
        # "4-year resale value" is ignored because it is a numeric feature with 36 nan values out of 157
        # "Model" is ignored because it is a categorical feature with 156 unique values out of 157
        # "Latest Launch" is ignored because it ranges in a small period thus it is likely not to be predictive
    }
    """The cars dataset features."""

    @staticmethod
    def load_data(filepath: str, full_features: bool, extrapolation: bool) -> AbstractManager.Data:
        """Loads the dataset.

        :param filepath:
            The dataset filepath.

        :param full_features:
            If True considers all the dataset features, otherwise considers only the monotonic ones.

        :param extrapolation:
            Whether to consider a test set with border samples or with random samples (interpolation).

        :return:
            A dictionary of dataframes representing the train and test sets, respectively.
        """
        df = pd.read_csv(filepath).replace({'.': np.nan})
        df = clean_dataframe(df, CarsManager.FEATURES)
        if full_features:
            df['manufacturer'] = df['manufacturer'].map(lambda v: v.strip())
            df = pd.get_dummies(df.dropna(), prefix_sep=': ')
        else:
            df = df[['price', 'sales']].dropna()
        extrapolation = {'price': 0.2} if extrapolation else None
        return split_dataset(df, extrapolation=extrapolation, test_size=0.2, val_size=0.0)

    def __init__(self, filepath: str, full_features: bool = False, full_grid: bool = False, extrapolation: bool = False,
                 grid_augmented: int = 150, grid_ground: Optional[int] = None, x_scaling: Method = 'std',
                 y_scaling: Method = 'norm', bound: Tuple[float, float] = (0, 100)):
        """
        :param filepath:
            The cars dataset filepath.

        :param full_features:
            Whether or not to use all the input features.

        :param full_grid:
            Whether or not to evaluate the results on an explicit full grid. This option is possible only if
            full_features is set to false, since there is no way to create an explicit grid on the full set of features.

        :param extrapolation:
            Whether or not to test on extrapolation instead of interpolation.

        :param grid_augmented:
            Number of augmented features for grid_kwargs. This will not have any effect if an explicit grid is passed.

        :param grid_ground:
            Number of ground samples for grid_kwargs. This will not have any effect if an explicit grid is passed.

        :param x_scaling:
            Scaling methods for the input data.

        :param y_scaling:
            Scaling methods for the output data.

        :param bound:
            The y bounds for the summary plot.
        """

        self.bound: Tuple[float, float] = bound
        """The y bounds for the summary plot."""

        # in case of full features, standardize floating features only (no skipped and no categorical features)
        grid = None
        if full_features:
            assert full_grid is False, "'full_grid' is not supported with 'full_features'"
            x_scaling = {v.alias or k: x_scaling for k, v in CarsManager.FEATURES.items() if v.dtype == 'float'}
        elif full_grid:
            grid = pd.DataFrame.from_dict({'price': np.linspace(self.bound[0], self.bound[1], 700)})
            x_scaling = {'price': x_scaling}
        super(CarsManager, self).__init__(
            directions={'price': -1},
            stratify=False,
            x_scaling=x_scaling,
            y_scaling=y_scaling,
            label='sales',
            loss=mean_squared_error,
            loss_name='mse',
            metric=r2_score,
            metric_name='r2',
            data_kwargs=dict(figsize=(14, 4), tight_layout=True),
            augmented_kwargs=dict(figsize=(10, 4)),
            summary_kwargs=dict(figsize=(10, 4), res=100, ylim=(-5, 125)),
            grid_kwargs=dict(num_augmented=grid_augmented, num_ground=grid_ground),
            grid=grid,
            filepath=filepath,
            full_features=full_features,
            extrapolation=extrapolation
        )

    def _get_sampling_functions(self, rng: Rng, num_augmented: Augmented = 15) -> SamplingFunctions:
        return {'price': (num_augmented, lambda s: rng.uniform(self.bound[0], self.bound[1], size=s))}

    def _data_plot(self, figsize: Figsize, tight_layout: TightLayout, **additional_kwargs):
        _, axes = plt.subplots(1, len(additional_kwargs), sharex='all', sharey='all',
                               figsize=figsize, tight_layout=tight_layout)
        for ax, (title, (x, y)) in zip(axes, additional_kwargs.items()):
            sns.scatterplot(x=x['price'], y=y, ax=ax).set(xlabel='price', ylabel='sales', title=title.capitalize())

    def _augmented_plot(self, aug: pd.DataFrame, figsize: Figsize, tight_layout: TightLayout, **additional_kwargs):
        plt.figure(figsize=figsize)
        sns.histplot(data=aug, x='price', hue='Augmented')

    def _summary_plot(self, model, figsize: Figsize, tight_layout: TightLayout, **additional_kwargs):
        res = additional_kwargs.pop('res')
        ylim = additional_kwargs.pop('ylim')
        plt.figure(figsize=figsize)
        for title, (x, y) in additional_kwargs.items():
            sns.scatterplot(x=x['price'], y=y, alpha=0.25, sizes=0.25, label=title.capitalize())
        x = pd.DataFrame(np.linspace(self.bound[0], self.bound[1], res), columns=['price'])
        y = model.predict(x).flatten()
        sns.lineplot(x=x['price'], y=y, color='black').set(ylabel='sales', title='Estimated Function')
        plt.xlim(self.bound)
        plt.ylim(ylim)
