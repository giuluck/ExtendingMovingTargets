import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, r2_score
from tensorflow.python.keras.utils.np_utils import to_categorical

from src.datasets.data_manager import DataManager


class RestaurantsManager(DataManager):
    @staticmethod
    def ctr_estimate(avg_ratings, num_reviews, dollar_ratings):
        dollar_rating_baseline = {'D': 3, 'DD': 2, 'DDD': 4, 'DDDD': 4.5}
        dollar_ratings = np.array([dollar_rating_baseline[d] for d in dollar_ratings])
        return 1 / (1 + np.exp(dollar_ratings - avg_ratings * np.log1p(num_reviews) / 4))

    @staticmethod
    def predict(dataframe):
        return RestaurantsManager.ctr_estimate(
            avg_ratings=dataframe['avg_rating'],
            num_reviews=dataframe['num_reviews'],
            dollar_ratings=dataframe[['D', 'DD', 'DDD', 'DDDD']].idxmax(axis=1)
        ).values

    @staticmethod
    def sample_restaurants(n, rng):
        avg_ratings = rng.uniform(1.0, 5.0, n)
        num_reviews = np.round(np.exp(rng.uniform(0.0, np.log(200), n)))
        dollar_ratings = rng.choice(['D', 'DD', 'DDD', 'DDDD'], n)
        ctr_labels = RestaurantsManager.ctr_estimate(avg_ratings, num_reviews, dollar_ratings)
        return avg_ratings, num_reviews, dollar_ratings, ctr_labels

    @staticmethod
    def sample_dataset(n, rng, testing_set=True):
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
        })

    @staticmethod
    def plot_conclusions(models, figsize=(10, 10), tight_layout=True, res=100, orient_columns=True):
        avg_ratings, num_reviews = np.meshgrid(np.linspace(1, 5, num=res), np.linspace(0, 200, num=res))
        rows, cols = (4, len(models)) if orient_columns else (len(models), 4)
        _, axes = plt.subplots(rows, cols, sharex='all', sharey='all', figsize=figsize, tight_layout=tight_layout)
        axes = axes.reshape((rows, cols))
        for i, rating in enumerate(['D', 'DD', 'DDD', 'DDDD']):
            df = pd.DataFrame({'avg_rating': avg_ratings.flatten(), 'num_reviews': num_reviews.flatten()})
            for d in ['D', 'DD', 'DDD', 'DDDD']:
                df[d] = 1 if rating == d else 0
            for j, (title, model) in enumerate(models.items()):
                ctr = model.predict(df)
                if orient_columns:
                    axes[i, j].pcolor(avg_ratings, num_reviews, ctr.reshape(res, res), shading='auto', vmin=0, vmax=1)
                    axes[i, j].set(title=title if i == 0 else None, xlabel=None, ylabel=rating if j == 0 else None)
                else:
                    axes[j, i].pcolor(avg_ratings, num_reviews, ctr.reshape(res, res), shading='auto', vmin=0, vmax=1)
                    axes[j, i].set(title=rating if j == 0 else None, xlabel=None, ylabel=title if i == 0 else None)
        plt.show()

    @staticmethod
    def process_data(dataset):
        for rating in ['DDDD', 'DDD', 'DD', 'D']:
            dataset.insert(2, rating, dataset['dollar_rating'] == rating)
        dataset = dataset.drop('dollar_rating', axis=1)
        dataset = dataset.astype({c: 'float' if c == 'avg_rating' else 'int' for c in dataset.columns})
        if 'clicked' in dataset.columns:
            return dataset.drop('clicked', axis=1), dataset['clicked']
        else:
            return dataset, RestaurantsManager.predict(dataset)

    def __init__(self, x_scaling: str = 'std', res: int = 40):
        ar, nr, dr = np.meshgrid(np.linspace(1, 5, num=res), np.linspace(0, 200, num=res), ['D', 'DD', 'DDD', 'DDDD'])
        grid, self.ground_truth = self.process_data(pd.DataFrame.from_dict({
            'avg_rating': ar.flatten(),
            'num_reviews': nr.flatten(),
            'dollar_rating': dr.flatten()
        }))
        super(RestaurantsManager, self).__init__(
            x_columns=['avg_rating', 'num_reviews', 'D', 'DD', 'DDD', 'DDDD'],
            x_scaling=dict(avg_rating=x_scaling, num_reviews=x_scaling),
            y_column='clicked',
            y_scaling=None,
            metric=roc_auc_score,
            metric_name='auc',
            grid=grid,
            data_kwargs=dict(figsize=(10, 8), tight_layout=True),
            augmented_kwargs=dict(figsize=(10, 4), tight_layout=True),
            summary_kwargs=dict(figsize=(14, 7), tight_layout=True, res=100)
        )

    def compute_ground_r2(self, model):
        pred = model.predict(self.grid)
        return r2_score(self.ground_truth, pred)

    def compute_monotonicities(self, samples, references, eps=1e-5):
        def categorical_monotonicities(diffs):
            mono = 1 * (diffs == 1) - 1 * (diffs == -1)  # DD (2) > D    (1) -> DD - D = 1, D - DD = -1
            mono += 1 * (diffs == -6) - 1 * (diffs == 6)  # DD (2) > DDDD (8) -> DD - DDDD = -6, DDDD - DD = 6
            return mono

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
        num_differences = np.sign(np.abs(differences)).sum(axis=0)
        # convert categorical differences to monotonicities and get whole monotonicity (sum of monotonicity signs)
        differences[-1] = categorical_monotonicities(differences[-1])
        monotonicities = np.sign(differences).sum(axis=0).transpose()
        # the final monotonicities are masked for pairs with just one different attribute
        monotonicities = monotonicities.astype('int') * (num_differences == 1)
        return monotonicities

    def _load_splits(self):
        rng = np.random.default_rng(seed=0)
        return {
            'train': self.process_data(self.sample_dataset(1000, rng, testing_set=False)),
            'validation': self.process_data(self.sample_dataset(600, rng, testing_set=False)),
            'test': self.process_data(self.sample_dataset(600, rng, testing_set=True))
        }

    def _get_sampling_functions(self, num_augmented, rng):
        dollar_rating = ('D', 'DD', 'DDD', 'DDDD')
        return {
            'avg_rating': (num_augmented // 3, lambda s: rng.uniform(1.0, 5.0, size=s)),
            'num_reviews': (num_augmented // 3, lambda s: np.round(np.exp(rng.uniform(0.0, np.log(200), size=s)))),
            dollar_rating: (num_augmented // 3, lambda s: to_categorical(rng.integers(4, size=s), num_classes=4))
        }

    def _data_plot(self, figsize, tight_layout, kind='distributions', **kwargs):
        _, ax = plt.subplots(len(kwargs), 3, sharex='col', figsize=figsize, tight_layout=tight_layout)
        ax = ax.reshape((-1, 3))
        for i, (title, (x, y)) in enumerate(kwargs.items()):
            sns.kdeplot(x=x['avg_rating'], hue=y, ax=ax[i, 0]).set(ylabel=title.capitalize())
            sns.kdeplot(x=x['num_reviews'], hue=y, ax=ax[i, 1])
            sns.countplot(x=x[['D', 'DD', 'DDD', 'DDDD']].idxmax(axis=1), hue=y, ax=ax[i, 2])

    # noinspection PyMethodOverriding
    def _augmented_plot(self, aug, figsize, tight_layout, **kwargs):
        _, axes = plt.subplots(1, 3, sharex='col', sharey='row', figsize=figsize, tight_layout=tight_layout)
        aug['dollar_rating'] = aug[['D', 'DD', 'DDD', 'DDDD']].idxmax(axis=1)
        for ax, feature in zip(axes, ['avg_rating', 'num_reviews', 'dollar_rating']):
            sns.histplot(data=aug, x=feature, hue='Augmented', ax=ax)

    # noinspection PyMethodOverriding
    def _summary_plot(self, model, res, figsize, tight_layout, **kwargs):
        print(f'{self.compute_ground_r2(model):.4} (ground r2)')
        self.plot_conclusions(figsize=figsize, tight_layout=tight_layout, res=res, orient_columns=False, models={
            'Estimated CTR': model,
            'Real CTR': RestaurantsManager
        })
