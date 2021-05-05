import numpy as np
import pandas as pd

from src.util.preprocessing import Scaler, split_dataset


def synthetic_function(a, b):
    a = a ** 3
    b = np.sin(np.pi * (b - 0.01)) ** 2 + 1
    return a / b + b


def load_synthetic(noise=0.0, extrapolation=False):
    # generate and split data
    rng = np.random.default_rng(seed=0)
    if extrapolation:
        df = pd.DataFrame.from_dict({
            'a': rng.uniform(low=-1, high=1, size=700),
            'b': rng.uniform(low=-1, high=1, size=700)
        })
        splits = split_dataset(df, extrapolation={'a': 0.7}, val_size=0.25, random_state=0)
    else:
        df = [
            {'a': rng.normal(scale=0.2, size=150).clip(min=-1, max=1), 'b': rng.uniform(low=-1, high=1, size=150)},
            {'a': rng.normal(scale=0.2, size=50).clip(min=-1, max=1), 'b': rng.uniform(low=-1, high=1, size=50)},
            {'a': rng.uniform(low=-1, high=1, size=500), 'b': rng.uniform(low=-1, high=1, size=500)}
        ]
        splits = {s: pd.DataFrame.from_dict(x) for s, x in zip(['train', 'validation', 'test'], df)}
    # assign y values
    splits = {s: (x, pd.Series(synthetic_function(x['a'], x['b']), name='label') + rng.normal(scale=noise, size=len(x)))
              for s, x in splits.items()}
    return get_output_dictionary(splits, x_method=(-1, 1), y_method='norm')


def load_puzzles(filepath, extrapolation=False):
    # preprocess data
    df = pd.read_csv(filepath)
    for col in df.columns:
        if col not in ['label', 'split']:
            df[col] = df[col].map(lambda l: l.strip('[]').split(';')).map(lambda l: [float(v.strip()) for v in l])
    x = pd.DataFrame()
    x['word_count'] = df['word_count'].map(lambda l: np.mean(l))
    x['star_rating'] = df['star_rating'].map(lambda l: np.mean(l))
    x['num_reviews'] = df['star_rating'].map(lambda l: len(l))
    y = df['label']
    # split data
    if extrapolation:
        splits = split_dataset(x, y, val_size=0.2, extrapolation=0.2)
    else:
        splits = {s: (x[df['split'] == s], y[df['split'] == s]) for s in ['train', 'validation', 'test']}
    return get_output_dictionary(splits, x_method='zeromax', y_method='zeromax')
