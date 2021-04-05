import numpy as np
import pandas as pd


def ctr_estimate(avg_ratings, num_reviews, dollar_ratings):
    dollar_rating_baseline = {'D': 3, 'DD': 2, 'DDD': 4, 'DDDD': 4.5}
    dollar_ratings = np.array([dollar_rating_baseline[d] for d in dollar_ratings])
    return 1 / (1 + np.exp(dollar_ratings - avg_ratings * np.log1p(num_reviews) / 4))


def sample_restaurants(n, rng):
    avg_ratings = rng.uniform(1.0, 5.0, n)
    num_reviews = np.round(np.exp(rng.uniform(0.0, np.log(200), n)))
    dollar_ratings = rng.choice(['D', 'DD', 'DDD', 'DDDD'], n)
    ctr_labels = ctr_estimate(avg_ratings, num_reviews, dollar_ratings)
    return avg_ratings, num_reviews, dollar_ratings, ctr_labels


def sample_dataset(n, rng, testing_set=True):
    (avg_ratings, num_reviews, dollar_ratings, ctr_labels) = sample_restaurants(n, rng)
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


def load_data():
    def process(ds):
        for rating in ['DDDD', 'DDD', 'DD', 'D']:
            ds.insert(2, rating, ds['dollar_rating'] == rating)
        ds = ds.drop('dollar_rating', axis=1)
        ds = ds.astype({c: 'float' if c == 'avg_rating' else 'int' for c in ds.columns})
        return ds.drop('clicked', axis=1), ds['clicked']

    rng = np.random.default_rng(seed=0)
    train_data = sample_dataset(1000, rng, testing_set=False)
    val_data = sample_dataset(600, rng, testing_set=False)
    test_data = sample_dataset(600, rng, testing_set=True)
    return process(train_data), process(val_data), process(test_data)
