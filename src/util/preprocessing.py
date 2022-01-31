from typing import Dict, List, Optional

import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from src.util.typing import Extrapolation


def split_dataset(*dataset,
                  test_size: float = 0.2,
                  val_size: Optional[float] = None,
                  extrapolation: Extrapolation = None,
                  random_state: int = 0,
                  shuffle: bool = True,
                  stratify: Optional = None) -> Dict:
    """Splits the input data."""
    val_size = test_size if val_size is None else val_size
    val_size = val_size if isinstance(val_size, float) else val_size / len(dataset[0])
    test_size = test_size if isinstance(test_size, float) else test_size / len(dataset[0])
    # split train/test
    if extrapolation is None:
        splits = train_test_split(*dataset, test_size=test_size, random_state=random_state,
                                  shuffle=shuffle, stratify=stratify)
    else:
        x = dataset[0]
        if not isinstance(extrapolation, dict):
            extrapolation = {col: extrapolation for col in x.columns}
        train_mask, test_mask = np.ones(len(x)).astype(bool), np.ones(len(x)).astype(bool)
        # removes all the data points at the borders for each feature (lq and uq are the quantile values for test set)
        for col, ex in extrapolation.items():
            feat = x[col]
            lq, uq = ex / 2, 1 - ex / 2 if isinstance(ex, float) else ex
            train_mask = np.logical_and(train_mask, np.logical_and(feat > feat.quantile(lq), feat < feat.quantile(uq)))
            test_mask = np.logical_and(test_mask, np.logical_or(feat <= feat.quantile(lq), feat >= feat.quantile(uq)))
        splits = []
        # create the splits from the initial data by appending the train and the test partition for each vector
        for d in dataset:
            splits.append(d[train_mask])
            splits.append(d[test_mask])
    train_data, test_data = splits if len(dataset) == 1 else (splits[::2], splits[1::2])
    # split val/test only if necessary
    if val_size == 0.0:
        return {'train': train_data, 'test': test_data}
    else:
        splits = train_test_split(*train_data, test_size=val_size, random_state=random_state,
                                  shuffle=shuffle, stratify=stratify)
        train_data, val_data = splits if len(dataset) == 1 else (splits[::2], splits[1::2])
        return {'train': train_data, 'validation': val_data, 'test': test_data}


def cross_validate(*dataset,
                   num_folds: int = 10,
                   random_state: int = 0,
                   shuffle: bool = True,
                   stratify: Optional = None) -> List[Dict]:
    """Splits the input data in folds."""
    kf = KFold if stratify is None else StratifiedKFold
    kf = kf(n_splits=num_folds, random_state=random_state, shuffle=shuffle)
    folds = []
    for tr, vl in kf.split(X=dataset[0], y=stratify):
        folds += [{'train': tuple([v.iloc[tr] for v in dataset]), 'validation': tuple([v.iloc[vl] for v in dataset])}]
    return folds
