import numpy as np
import pandas as pd

from src.util.preprocessing import Scaler, split_dataset


def load_cars(filepath):
    # preprocess and split data
    df = pd.read_csv(filepath).rename(columns={'Price in thousands': 'price', 'Sales in thousands': 'sales'})
    df = df[['price', 'sales']].replace({'.': np.nan}).dropna().astype('float')
    splits = split_dataset(df[['price']], df['sales'], shuffle=False)
    # configure scalers and rescale
    xsc = Scaler(splits[0][0], 'zeromax')
    ysc = Scaler(splits[0][1], 'zeromax')
    splits = [(xsc.transform(x).reset_index(drop=True), ysc.transform(y).reset_index(drop=True)) for x, y in splits]
    # get dictionary
    splits = {k: v for k, v in zip(['train', 'validation', 'test'], splits)}
    splits['scalers'] = (xsc, ysc)
    return splits


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
