import numpy as np
import pandas as pd

from src.util.preprocessing import Scaler, split_dataset


def synthetic_function(a, b):
    a = 2 * ((a + 0.01) ** 3)
    b = 4 * ((b - 0.01) ** 2)
    return a / (b + 1) + b


def load_synthetic():
    # assign number of samples and samples distributions to each split
    rng = np.random.default_rng(seed=0)
    datasets = {
        'train': (200, lambda s: rng.normal(scale=0.2, size=s)),
        'validation': (100, lambda s: rng.normal(scale=0.2, size=s)),
        'test': (500, lambda s: 2 * rng.uniform(size=s) - 1),
    }
    # generate samples
    for key, (size, get_samples) in datasets.items():
        x = pd.DataFrame.from_dict({'a': get_samples(size), 'b': 2 * rng.uniform(size=size) - 1})
        y = pd.Series(synthetic_function(x['a'], x['b']), name='label')
        datasets[key] = (x + rng.normal(scale=[0.05, 0.1], size=x.shape), y + rng.normal(scale=0.5, size=y.shape))
    # configure scalers and rescale
    xsc, ysc = Scaler(datasets['train'][0], (-1, 1)), Scaler(datasets['train'][1], (-1, 1))
    outputs = {key: (xsc.transform(x), ysc.transform(y)) for key, (x, y) in datasets.items()}
    outputs['scalers'] = (xsc, ysc)
    return outputs


def load_cars(filepath):
    # preprocess and split data
    df = pd.read_csv(filepath).rename(columns={'Price in thousands': 'price', 'Sales in thousands': 'sales'})
    df = df[['price', 'sales']].replace({'.': np.nan}).dropna().astype('float')
    splits = split_dataset(df[['price']], df['sales'], random_state=0)
    # configure scalers and rescale
    xsc = Scaler(splits[0][0], 'zeromax')
    ysc = Scaler(splits[0][1], 'zeromax')
    splits = [(xsc.transform(x).reset_index(drop=True), ysc.transform(y).reset_index(drop=True)) for x, y in splits]
    # get dictionary
    outputs = {k: v for k, v in zip(['train', 'validation', 'test'], splits)}
    outputs['scalers'] = (xsc, ysc)
    return outputs


def load_puzzles(filepath):
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
    # configure scalers
    x_scaler = Scaler(x[df['split'] == 'train'], 'zeromax')
    y_scaler = Scaler(y[df['split'] == 'train'], 'zeromax')
    # split data
    outputs = {}
    for split in ['train', 'validation', 'test']:
        split_x = x[df['split'] == split].reset_index(drop=True)
        split_y = y[df['split'] == split].reset_index(drop=True)
        outputs[split] = (x_scaler.transform(split_x), y_scaler.transform(split_y))
    outputs['scalers'] = (x_scaler, y_scaler)
    return outputs
