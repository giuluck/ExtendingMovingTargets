import pandas as pd

def load_data(filepath, split='tuple', **kwargs):
    df = pd.read_csv(filepath, **kwargs)
    if split is None or split == 'none':
        return df
    dfs = {key: df[df['split'] == key].drop('split', axis=1).reset_index(drop=True) for key in ['train', 'val', 'test']}
    if split == 'dict' or split == 'dictionary':
        return dfs
    if split == 'tuple':
        return dfs['train'], dfs['val'], dfs['test']
    raise ValueError(f'{split} is not a valid argument')
