import numpy as np
import pandas as pd

from src.util.preprocessing import Scaler, split_dataset

SPLITS = ['train', 'validation', 'test']


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
        splits = [pd.DataFrame.from_dict(x) for x in df]
    # assign y values
    splits = [(x, pd.Series(synthetic_function(x['a'], x['b']), name='label') + rng.normal(scale=noise, size=len(x)))
              for x in splits]
    return get_output_dictionary(splits, x_method=(-1, 1), y_method='norm')


def load_cars(filepath, extrapolation=False):
    # preprocess data
    df = pd.read_csv(filepath).rename(columns={'Price in thousands': 'price', 'Sales in thousands': 'sales'})
    df = df[['price', 'sales']].replace({'.': np.nan}).dropna().astype('float')
    # split data
    if extrapolation:
        splits = split_dataset(df[['price']], df['sales'], extrapolation=0.2, val_size=0.2, random_state=0)
    else:
        splits = split_dataset(df[['price']], df['sales'], extrapolation=None, test_size=0.2, random_state=0)
    return get_output_dictionary(splits, x_method='zeromax', y_method='zeromax')


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
        splits = [(x[df['split'] == s], y[df['split'] == s]) for s in ['train', 'validation', 'test']]
    return get_output_dictionary(splits, x_method='zeromax', y_method='zeromax')


def get_output_dictionary(splits, x_method, y_method):
    x_scaler, y_scaler = Scaler(splits[0][0], x_method), Scaler(splits[0][1], y_method)
    outputs = {s: (x_scaler.transform(x).reset_index(drop=True), y_scaler.transform(y).reset_index(drop=True))
               for s, (x, y) in zip(['train', 'validation', 'test'], splits)}
    outputs['scalers'] = (x_scaler, y_scaler)
    return outputs


if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt
    from src.regressions.plot import plot_cars, plot_puzzles, plot_synthetic

    ds = 'S'

    if ds == 'S':
        data = load_synthetic(extrapolation=False)
        plot_synthetic(**data)
        _, axes = plt.subplots(3, 2, sharex='col', sharey='all', tight_layout=True, figsize=(14, 12))
        for ii, ss in enumerate(['train', 'validation', 'test']):
            sns.histplot(data=data[ss][0], x='a', ax=axes[ii, 0]).set(title=f'a - {ss}')
            sns.histplot(data=data[ss][0], x='b', ax=axes[ii, 1]).set(title=f'b - {ss}')
        plt.show()

        data = load_synthetic(extrapolation=True)
        plot_synthetic(**data)
        _, axes = plt.subplots(3, 2, sharex='col', sharey='all', tight_layout=True, figsize=(14, 12))
        for ii, ss in enumerate(['train', 'validation', 'test']):
            sns.histplot(data=data[ss][0], x='a', ax=axes[ii, 0]).set(title=f'a - {ss}')
            sns.histplot(data=data[ss][0], x='b', ax=axes[ii, 1]).set(title=f'b - {ss}')
        plt.show()
    elif ds == 'C':
        data = load_cars('../../res/cars.csv', extrapolation=False)
        plot_cars(**data)
        data = load_cars('../../res/cars.csv', extrapolation=True)
        plot_cars(**data)
    elif ds == 'P':
        data = load_puzzles('../../res/puzzles.csv', extrapolation=False)
        plot_puzzles(**data)
        data = load_puzzles('../../res/puzzles.csv', extrapolation=True)
        plot_puzzles(**data)
