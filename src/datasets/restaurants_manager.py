"""Restaurants Data Manager."""

from typing import Tuple, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, log_loss
from tensorflow.python.keras.utils.np_utils import to_categorical

from src.datasets.abstract_manager import AbstractManager
from src.util.typing import Rng, Figsize, TightLayout, Augmented, SamplingFunctions, MonotonicitiesMatrix


class RestaurantsManager(AbstractManager):
    """Data Manager for the Restaurant Dataset."""

    @staticmethod
    def ctr_estimate(avg_ratings, num_reviews, dollar_ratings) -> Any:
        """Computes the real click-trough rate estimate.

        :param avg_ratings:
            Input vector of average ratings.

        :param num_reviews:
            Input vector of num reviews.

        :param dollar_ratings:
            Input vector of dollar ratings.

        :return:
            Output vector of click-through rate values.
        """
        dollar_rating_baseline = {'D': 3, 'DD': 2, 'DDD': 4, 'DDDD': 4.5}
        dollar_ratings = np.array([dollar_rating_baseline[d] for d in dollar_ratings])
        return 1 / (1 + np.exp(dollar_ratings - avg_ratings * np.log1p(num_reviews) / 4))

    @staticmethod
    def predict(dataframe: pd.DataFrame) -> np.ndarray:
        """Predicts the real click-trough rate estimate from a dataframe.

        :param dataframe:
            The input data.

        :return:
            The output vector predicted click-through rates.
        """
        return RestaurantsManager.ctr_estimate(
            avg_ratings=dataframe['avg_rating'],
            num_reviews=dataframe['num_reviews'],
            dollar_ratings=dataframe[['D', 'DD', 'DDD', 'DDDD']].idxmax(axis=1)
        ).values

    @staticmethod
    def sample_restaurants(n: int, rng: Rng) -> Tuple[Any, Any, Any, Any]:
        """Samples ground truth about restaurants.

        :param n:
            The number of samples.

        :param rng:
            A random number generator.

        :return:
            A tuple with four vectors, each one representing either one of the three input features or the output
            feature of the restaurants dataset.
        """
        avg_ratings = rng.uniform(1.0, 5.0, n)
        num_reviews = np.round(np.exp(rng.uniform(0.0, np.log(200), n)))
        dollar_ratings = rng.choice(['D', 'DD', 'DDD', 'DDDD'], n)
        ctr_labels = RestaurantsManager.ctr_estimate(avg_ratings, num_reviews, dollar_ratings)
        return avg_ratings, num_reviews, dollar_ratings, ctr_labels

    @staticmethod
    def sample_dataset(n: int, rng: Rng, testing_set: bool = True) -> pd.DataFrame:
        """Sample data points from the restaurants.

        :param n:
            The number of restaurant samples.

        :param rng:
            A random number generator.

        :param testing_set:
            Whether or not the restaurants must be sampled according to the test set distribution.

        :return:
            A `DataFrame` containing the three input features and the output one.
        """
        (avg_ratings, num_reviews, dollar_ratings, ctr_labels) = RestaurantsManager.sample_restaurants(n, rng)
        # testing has a more uniform distribution over all restaurants
        # while training/validation datasets have more views on popular restaurants
        if testing_set:
            num_views = rng.poisson(lam=3, size=n)
        else:
            num_views = rng.poisson(lam=ctr_labels * num_reviews / 50.0, size=n)
        return pd.DataFrame({
            'avg_rating': np.repeat(avg_ratings, num_views),
            'num_reviews': np.repeat(num_reviews, num_views),
            'dollar_rating': np.repeat(dollar_ratings, num_views),
            'clicked': rng.binomial(n=1, p=np.repeat(ctr_labels, num_views))
        }).astype({'clicked': int})

    # noinspection DuplicatedCode
    @staticmethod
    def plot_conclusions(models, figsize: Figsize = (10, 10), tight_layout: TightLayout = True,
                         res: int = 100, orient_columns: bool = True, show: bool = True):
        """Plots the conclusions by comparing different models.

        :param models:
            A list of model objects having the 'predict(x)' method.

        :param figsize:
            The figsize parameter passed to `plt()`.

        :param tight_layout:
            The tight_layout parameter passed to `plt()`.

        :param res:
            The evaluation grid resolution.

        :param orient_columns:
            Whether to compare the models by column or by row.

        :param show:
            Whether or not to show the final plot.
        """
        avg_ratings, num_reviews = np.meshgrid(np.linspace(1, 5, num=res), np.linspace(0, 200, num=res))
        rows, cols = (4, len(models)) if orient_columns else (len(models), 4)
        _, axes = plt.subplots(rows, cols, sharex='all', sharey='all', figsize=figsize, tight_layout=tight_layout)
        axes = axes.reshape((rows, cols))
        for i, rating in enumerate(['D', 'DD', 'DDD', 'DDDD']):
            df = pd.DataFrame({'avg_rating': avg_ratings.flatten(), 'num_reviews': num_reviews.flatten()})
            for d in ['D', 'DD', 'DDD', 'DDDD']:
                df[d] = 1 if rating == d else 0
            for j, (title, model) in enumerate(models.items()):
                ctr = model.predict(df).reshape(res, res)
                if orient_columns:
                    axes[i, j].pcolor(avg_ratings, num_reviews, ctr, shading='auto', cmap='viridis', vmin=0, vmax=1)
                    axes[i, j].set(title=title if i == 0 else None, xlabel=None, ylabel=rating if j == 0 else None)
                else:
                    axes[j, i].pcolor(avg_ratings, num_reviews, ctr, shading='auto', cmap='viridis', vmin=0, vmax=1)
                    axes[j, i].set(title=rating if j == 0 else None, xlabel=None, ylabel=title if i == 0 else None)
        if show:
            plt.show()

    @staticmethod
    def process_data(dataset: pd.DataFrame) -> Tuple[Any, np.ndarray]:
        """Processes the dataframe by applying one-hot encoding and computing the ground truths if needed.

        :param dataset:
            A `DataFrame` for restaurants dataset.

        :return:
            A tuple containing the same dataset with one-hot encoding for categorical value in the first position, and
            a `Series` containing the ground truths in the second position.
        """
        for rating in ['DDDD', 'DDD', 'DD', 'D']:
            dataset.insert(2, rating, dataset['dollar_rating'] == rating)
        dataset = dataset.drop('dollar_rating', axis=1)
        dataset = dataset.astype({c: 'float' if c == 'avg_rating' else 'int' for c in dataset.columns})
        if 'clicked' in dataset.columns:
            return dataset.drop('clicked', axis=1), dataset['clicked']
        else:
            return dataset, RestaurantsManager.predict(dataset)

    @staticmethod
    def load_data() -> AbstractManager.Data:
        """Loads the dataset.

        :return:
            A dictionary of dataframes representing the train and test sets, respectively.
        """
        rng = np.random.default_rng(seed=0)
        splits = {
            'train': RestaurantsManager.process_data(RestaurantsManager.sample_dataset(1000, rng, testing_set=False)),
            'test': RestaurantsManager.process_data(RestaurantsManager.sample_dataset(600, rng, testing_set=True))
        }
        return {split: pd.concat(data, axis=1) for split, data in splits.items()}

    def __init__(self, full_grid: bool = False, grid_augmented: int = 10,
                 grid_ground: Optional[int] = None, x_scaling: str = 'std'):
        """
        :param full_grid:
            Whether or not to evaluate the results on an explicit full grid. This option is possible only if
            full_features is set to false, since there is no way to create an explicit grid on the full set of features.

        :param grid_augmented:
            Number of augmented features for grid_kwargs. This will not have any effect if an explicit grid is passed.

        :param grid_ground:
            Number of ground samples for grid_kwargs. This will not have any effect if an explicit grid is passed.

        :param x_scaling:
            Scaling methods for the input data.
        """

        self.ground_truth = None
        """The vector of ground truths for the explicit grid, if present."""

        if full_grid:
            ar, nr, dr = np.meshgrid(np.linspace(1, 5, num=40), np.linspace(0, 200, num=40), ['D', 'DD', 'DDD', 'DDDD'])
            grid, self.ground_truth = self.process_data(pd.DataFrame.from_dict({
                'avg_rating': ar.flatten(),
                'num_reviews': nr.flatten(),
                'dollar_rating': dr.flatten()
            }))
        else:
            grid = None
        super(RestaurantsManager, self).__init__(
            directions={'avg_rating': 1, 'num_reviews': 1},
            stratify=True,
            x_scaling=dict(avg_rating=x_scaling, num_reviews=x_scaling),
            y_scaling=None,
            label='clicked',
            loss=log_loss,
            loss_name='bce',
            metric=roc_auc_score,
            metric_name='auc',
            data_kwargs=dict(figsize=(10, 8), tight_layout=True),
            augmented_kwargs=dict(figsize=(10, 4), tight_layout=True),
            summary_kwargs=dict(figsize=(14, 7), tight_layout=True, res=100),
            grid_kwargs=dict(num_augmented=grid_augmented, num_ground=grid_ground),
            grid=grid
        )

    def compute_monotonicities(self,
                               samples: AbstractManager.Samples,
                               references: AbstractManager.Samples,
                               eps: float = 1e-5) -> MonotonicitiesMatrix:
        """Routine to compute restaurants dataset monotonicities involving categorical pairwise monotonicities
        regarding the 'dollar_rating' attribute.

        Indeed, categorical monotonicities in this domain are so that we can expect category 'DD' to have a greater
        click-through rate with respect to category 'D' and to category 'DDDD' as well, all else being equal.

        :param samples:
            The data samples.

        :param references:
            The reference samples.

        :param eps:
            The slack value under which a violation is considered to be acceptable.

        :return:
            A NxM matrix where N is the number of samples and M is the number of references, where each cell is filled
            with -1, 0, or 1 depending on the kind of monotonicity between samples[i] and references[j].
        """

        def _categorical_monotonicities(diffs):
            mono = 1 * (diffs == 1) - 1 * (diffs == -1)  # DD (2) > D    (1) -> DD - D = 1, D - DD = -1
            mono += 1 * (diffs == -6) - 1 * (diffs == 6)  # DD (2) > DDDD (8) -> DD - DDDD = -6, DDDD - DD = 6
            return mono

        # check input data
        columns = ['avg_rating', 'num_reviews', 'D', 'DD', 'DDD', 'DDDD']
        if isinstance(references, pd.Series):
            references = pd.DataFrame(references).transpose()
        assert list(samples.columns) == columns, f"samples columns {list(samples.columns)} are not supported"
        assert list(references.columns) == columns, f"references column {list(references.columns)} are not supported"
        # convert dataframes/series into a matrices
        samples, references = samples[columns].values, references[columns].values
        # transpose tensors to get shape (6, ...)
        samples, references = samples.copy().transpose(), references.copy().transpose()
        # store categorical values into single element (D -> 1, DD -> 2, DDD -> 4, DDDD -> 8)
        samples[2] = 2 ** samples[2:6].argmax(axis=0)
        references[2] = 2 ** references[2:6].argmax(axis=0)
        # transpose back with single categorical feature (..., 3) and increase samples dimension to match references
        samples, references = samples[:3].transpose(), references[:3].transpose()
        samples = np.hstack([samples] * len(references)).reshape((-1, len(references), 3))
        # compute differences between samples to get the number of different attributes
        differences = (samples - references).transpose()
        differences[np.abs(differences) < eps] = 0.
        num_differences = np.sign(np.abs(differences)).sum(axis=0).transpose()
        # convert categorical differences to monotonicities and get whole monotonicity (sum of monotonicity signs)
        differences[-1] = _categorical_monotonicities(differences[-1])
        monotonicities = np.sign(differences).sum(axis=0).transpose()
        monotonicities = np.squeeze(monotonicities * (num_differences == 1)).astype('int')
        # if a there is a single sample and a single reference, numpy.sum(axis=-1) will return a zero-dimensional array
        # instead of a scalar, thus it is necessary to manually handle this case
        return np.int32(monotonicities) if monotonicities.ndim == 0 else monotonicities

    def _get_sampling_functions(self, rng: Rng, num_augmented: Augmented = 5) -> SamplingFunctions:
        dollar_rating = ('D', 'DD', 'DDD', 'DDDD')
        return {
            'avg_rating': (num_augmented, lambda s: rng.uniform(1.0, 5.0, size=s)),
            'num_reviews': (num_augmented, lambda s: np.round(np.exp(rng.uniform(0.0, np.log(200), size=s)))),
            dollar_rating: (num_augmented, lambda s: to_categorical(rng.integers(4, size=s), num_classes=4))
        }

    def _data_plot(self, figsize: Figsize, tight_layout: TightLayout, **additional_kwargs):
        _, ax = plt.subplots(len(additional_kwargs), 3, sharex='col', figsize=figsize, tight_layout=tight_layout)
        ax = ax.reshape((-1, 3))
        for i, (title, (x, y)) in enumerate(additional_kwargs.items()):
            sns.kdeplot(x=x['avg_rating'], hue=y, ax=ax[i, 0]).set(ylabel=title.capitalize())
            sns.kdeplot(x=x['num_reviews'], hue=y, ax=ax[i, 1])
            sns.countplot(x=x[['D', 'DD', 'DDD', 'DDDD']].idxmax(axis=1), hue=y, ax=ax[i, 2])

    def _augmented_plot(self, aug: pd.DataFrame, figsize: Figsize, tight_layout: TightLayout, **additional_kwargs):
        _, axes = plt.subplots(1, 3, sharex='col', sharey='row', figsize=figsize, tight_layout=tight_layout)
        aug['dollar_rating'] = aug[['D', 'DD', 'DDD', 'DDDD']].idxmax(axis=1)
        for ax, feature in zip(axes, ['avg_rating', 'num_reviews', 'dollar_rating']):
            sns.histplot(data=aug, x=feature, hue='Augmented', ax=ax)

    def _summary_plot(self, model, figsize: Figsize, tight_layout: TightLayout, **additional_kwargs):
        self.plot_conclusions(models={'Estimated CTR': model, 'Real CTR': RestaurantsManager}, figsize=figsize,
                              tight_layout=tight_layout, res=additional_kwargs.pop('res'), orient_columns=False,
                              show=False)
