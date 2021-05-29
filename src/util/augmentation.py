"""Augmentation utils."""

from typing import Callable, Optional

import numpy as np
import pandas as pd

from moving_targets.util.typing import Matrix, Vector, Monotonicities
from src.util.typing import SamplingFunctions, AugmentedData, Directions


def augment_data(x: Matrix,
                 y: Vector,
                 compute_monotonicities: Callable,
                 sampling_functions: SamplingFunctions) -> AugmentedData:
    """Augments the dataset using the given sampling functions, then computes its monotonicities wrt the ground samples.

    Args:
        x: the matrix/dataframe of samples.
        y: the vector of labels.
        compute_monotonicities: routine that computes the monotonicities of some data points wrt a reference point.
        sampling_functions: sampling routine associated to each x feature, with respective number of augmented samples.

    Returns:
        A tuple of augmented inputs/labels. The labels are no more in form of a vector, but rather a `DataFrame` with
        two additional features: 'ground_index' and 'monotonicity' (wrt the ground index).
    """
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
    x_aug = pd.concat([x] + new_samples).reset_index(drop=True).astype(x.dtypes)
    y_aug = pd.concat([pd.DataFrame(y)] + new_info).reset_index(drop=True).rename({0: y.name}, axis=1)
    # there is no way to return integer labels due to the presence of nan values (pd.Int64 type have problems with tf)
    y_aug = y_aug.fillna({'ground_index': pd.Series(y.index), 'monotonicity': 0}).astype({'ground_index': int})
    return x_aug, y_aug


def compute_numeric_monotonicities(samples: np.ndarray,
                                   references: np.ndarray,
                                   directions: Directions,
                                   eps: float = 1e-6) -> np.ndarray:
    """Default way of computing monotonicities in case all the features are numeric.

    Args:
        samples: the matrix of data points.
        references: the vector/matrix of reference data point(s).
        directions: the direction of the monotonicity wrt each feature.
        eps: the slack value under which a violation is considered to be acceptable

    Returns:
        A NxM matrix where N is the number of samples and M is the number of references, where each cell is filled with
        -1, 0, or 1 depending on the kind of monotonicity between samples[i] and references[j].
    """
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


def get_monotonicities_list(data: pd.DataFrame,
                            kind: str = 'group',
                            label: Optional[str] = None,
                            compute_monotonicities: Callable = None,
                            errors: str = 'raise') -> Monotonicities:
    """Computes the monotonicities list from an augmented dataframe.

    Args:
        data: the augmented dataframe.
        kind: the monotonicity computation modality:
              - 'ground', which computes the monotonicity within each subgroup respectively to the ground index only.
              - 'group', which computes the monotonicity within each subgroup between each pair in the subgroup.
              - 'all', which computes the monotonicity between each pair in the whole dataset (very slow).
        label: the y label to be dropped, if present.
        compute_monotonicities: routine that computes the monotonicities of some data points wrt a reference point.
        errors: pandas argument that sets the behaviour in case a non present column is being dropped.

    Returns:
        A list of tuples where the first element is the higher index and the second element is the lower one.
    """
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
    else:
        raise ValueError("kind should be in ['ground', 'group', 'all']")
    return [(hi, li) for hi, li in zip(higher_indices, lower_indices)]
