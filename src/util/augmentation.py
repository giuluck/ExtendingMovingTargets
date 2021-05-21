from typing import Callable, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd


def augment_data(x, y, compute_monotonicities: Callable, sampling_functions: Dict[str, Tuple[int, Callable]]):
    new_samples = []
    new_info = []
    for ground_index, sample in x.iterrows():
        for attribute, (num_augmented, function) in sampling_functions.items():
            if num_augmented > 0:
                samples = pd.DataFrame([sample] * num_augmented)
                # handle monotonicities on multiple attribute with tuples (python dict do not allow lists as key)
                if isinstance(attribute, tuple):
                    attribute = list(attribute)
                samples[attribute] = function(num_augmented)
                monotonicities = compute_monotonicities(samples.values, sample.values.reshape(1, -1)).reshape(-1, )
                new_samples.append(samples.astype(x.dtypes))
                new_info.append(pd.DataFrame(
                    data=zip([ground_index] * num_augmented, monotonicities),
                    columns=['ground_index', 'monotonicity'],
                    dtype='int'
                ))
            else:
                new_samples.append(x.head(0))
                new_info.append(pd.DataFrame(data=[], columns=['ground_index', 'monotonicity'], dtype='int'))
    x_aug = pd.concat([x] + new_samples).reset_index(drop=True)
    y_aug = pd.concat([y] + new_info).reset_index(drop=True).rename({0: y.name}, axis=1)
    y_aug = y_aug.fillna({'ground_index': pd.Series(y_aug.index), 'monotonicity': 0}).astype({'ground_index': 'int'})
    return x_aug, y_aug


def compute_numeric_monotonicities(samples: np.ndarray, references: np.ndarray, directions: object, eps: float = 1e-5):
    # increase samples dimension to match references
    samples = np.hstack([samples] * len(references)).reshape((len(samples), len(references), -1))
    # compute differences between samples to get the number of different attributes
    differences = samples - references
    differences[np.abs(differences) < eps] = 0.
    num_differences = np.sign(np.abs(differences)).sum(axis=-1)
    # get whole monotonicity (sum of monotonicity signs) and mask for pairs with just one different attribute
    # directions is either a float or an array that says whether the monotonicity is increasing or decreasing (or null)
    monotonicities = np.sign(directions * differences).sum(axis=-1)
    monotonicities = monotonicities.astype('int') * (num_differences == 1)
    return monotonicities


def get_monotonicities_list(data: pd.DataFrame, kind: str, label: Optional[str],
                            compute_monotonicities: Callable = None, errors: str = 'raise'):
    higher_indices, lower_indices = [], []
    # ground monotonicities: retrieved from the 'monotonicity' field of the dataframe
    if kind == 'ground':
        for idx, rec in data.iterrows():
            monotonicity, ground_index = rec['monotonicity'], int(rec['ground_index'])
            if monotonicity != 0:
                higher_indices.append(idx if monotonicity > 0 else ground_index)
                lower_indices.append(ground_index if monotonicity > 0 else idx)
    # group monotonicities: retrieved using the 'compute_monotonicities' function, grouping by ground index
    elif kind == 'group':
        for index, group in data.groupby('ground_index'):
            values = group.drop([label, 'ground_index', 'monotonicity'], errors=errors, axis=1).values
            his, lis = np.where(compute_monotonicities(values, values) == 1)
            higher_indices.append(group.index.values[his])
            lower_indices.append(group.index.values[lis])
        higher_indices = np.concatenate(higher_indices)
        lower_indices = np.concatenate(lower_indices)
    # all monotonicities: retrieved using the 'compute_monotonicities' function independently from the ground index
    elif kind == 'all':
        values = data.drop([label, 'ground_index', 'monotonicity'], errors=errors, axis=1).values
        higher_indices, lower_indices = np.where(compute_monotonicities(values, values) == 1)
    return [(hi, li) for hi, li in zip(higher_indices, lower_indices)]
