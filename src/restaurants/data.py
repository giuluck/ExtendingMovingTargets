import numpy as np
import pandas as pd


def ctr_estimate(avg_ratings, num_reviews, dollar_ratings):
    dollar_rating_baseline = {'D': 3, 'DD': 2, 'DDD': 4, 'DDDD': 4.5}
    dollar_ratings = np.array([dollar_rating_baseline[d] for d in dollar_ratings])
    return 1 / (1 + np.exp(dollar_ratings - avg_ratings * np.log1p(num_reviews) / 4))


def sample_restaurants(n):
    avg_ratings = np.random.uniform(1.0, 5.0, n)
    num_reviews = np.round(np.exp(np.random.uniform(0.0, np.log(200), n)))
    dollar_ratings = np.random.choice(['D', 'DD', 'DDD', 'DDDD'], n)
    ctr_labels = ctr_estimate(avg_ratings, num_reviews, dollar_ratings)
    return avg_ratings, num_reviews, dollar_ratings, ctr_labels


def sample_dataset(n, testing_set):
    (avg_ratings, num_reviews, dollar_ratings, ctr_labels) = sample_restaurants(n)
    # testing has a more uniform distribution over all restaurants
    # while training/validation datasets have more views on popular restaurants
    if testing_set:
        num_views = np.random.poisson(lam=3, size=n)
    else:
        num_views = np.random.poisson(lam=ctr_labels * num_reviews / 50.0, size=n)
    return pd.DataFrame({
        'avg_rating': np.repeat(avg_ratings, num_views),
        'num_reviews': np.repeat(num_reviews, num_views),
        'dollar_rating': np.repeat(dollar_ratings, num_views),
        'clicked': np.random.binomial(n=1, p=np.repeat(ctr_labels, num_views))
    }).astype({'avg_rating': 'float', 'num_reviews': 'int', 'dollar_rating': 'category', 'clicked': 'int'})


def augment_data(df, n=5):
    def monotonicity(x1, x2, att):
        x1 = x1[att]
        x2 = x2[att]
        if att != 'dollar_rating':
            return np.sign(x2 - x1).astype('int')
        elif (x2 == 'DD') and (x1 == 'D' or x1 == 'DDDD'):
            return 1
        elif (x1 == 'DD') and (x2 == 'D' or x2 == 'DDDD'):
            return -1
        else:
            return 0

    new_samples = []
    for _, x in df.iterrows():
        new_values = {
            'avg_rating': np.random.uniform(1.0, 5.0, n),
            'num_reviews': np.round(np.exp(np.random.uniform(0.0, np.log(200), n))),
            'dollar_rating': np.random.choice(['D', 'DD', 'DDD', 'DDDD'], n)
        }
        for attribute, values in new_values.items():
            samples = pd.DataFrame([x] * n)
            samples[attribute] = values
            samples['monotonicity'] = [monotonicity(x, s, attribute) for _, s in samples.iterrows()]
            samples['clicked'] = -1
            new_samples.append(samples.astype(df.dtypes))
    return pd.concat(new_samples).reset_index()
