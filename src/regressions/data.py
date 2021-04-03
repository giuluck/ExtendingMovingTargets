import numpy as np
import pandas as pd

from src.util.preprocessing import Scaler


def split_data(df, x_columns, y_columns, x_scaler, y_scaler):
    outputs = []
    for split in ['train', 'validation', 'test']:
        split_df = df[df['split'] == split].reset_index(drop=True)
        outputs.append((x_scaler.transform(split_df[x_columns]), y_scaler.transform(split_df[y_columns])))
    outputs.append((x_scaler, y_scaler))
    return tuple(outputs)


def load_cars(folder='./'):
    # preprocess data
    df = pd.read_csv(folder + 'cars.csv').rename(columns={'Price in thousands': 'price', 'Sales in thousands': 'sales'})
    df = df[['price', 'sales', 'split']].replace({'.': np.nan}).dropna().astype({'price': 'float', 'sales': 'float'})
    # configure scalers
    x_scaler = Scaler(df[df['split'] == 'train']['price'], 'norm')
    y_scaler = Scaler(df[df['split'] == 'train']['sales'], 'norm')
    # split data
    return split_data(df, 'price', 'sales', x_scaler, y_scaler)


def load_puzzles(folder='./'):
    # preprocess data
    df = pd.read_csv(folder + 'puzzles.csv')
    for col in df.columns:
        if col not in ['label', 'split']:
            df[col] = df[col].map(lambda l: l.strip('[]').split(';')).map(lambda l: [float(v.strip()) for v in l])
    df['word_count'] = df['word_count'].map(lambda l: np.mean(l))
    df['star_rating'] = df['star_rating'].map(lambda l: np.mean(l))
    df['num_reviews'] = df['is_amazon'].map(lambda l: len(l))
    # configure scalers
    x_scaler = Scaler(df[df['split'] == 'train'][['word_count', 'star_rating', 'num_reviews']], 'norm')
    y_scaler = Scaler(df[df['split'] == 'train']['label'], 'norm')
    # split data
    return split_data(df, ['word_count', 'star_rating', 'num_reviews'], 'label', x_scaler, y_scaler)
