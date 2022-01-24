"""Augmentation utils."""

from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.util.typing import MonotonicitiesList, MonotonicitiesMatrix
from src.util.typing import SamplingFunctions, AugmentedData


def augment_data(x: pd.DataFrame,
                 y: pd.Series,
                 sampling_functions: SamplingFunctions,
                 compute_monotonicities: Optional[Callable] = None) -> AugmentedData:
    """Augments the dataset using the given sampling functions, then computes its monotonicities wrt the ground samples.

    :param x:
        The matrix/dataframe of samples.

    :param y:
        The vector of labels.

    :param sampling_functions:
        Sampling routine associated to each x feature, with respective number of augmented samples, it must be a
        dictionary which associates to each string key representing the attribute a tuple
        (<number_of_augmented_samples>, <callable_sampling_function>), where <callable_sampling_function> is a function
        f(<n>) -> {x_1, ..., x_n} which returns <n> scalar values sampled from the data distribution related to the
        specified attribute.

    :param compute_monotonicities:
        Routine that computes the monotonicities of some data points wrt a reference point, it must be a function
        f(<x>, <y>) -> <monotonicities_matrix> that computes pairwise expected monotonicities between two vectors <x>
        and <y> (with size <n> and <m>, respectively) and stores them in a matrix having shape (<n>, <m>).

    :return:
        A tuple of augmented inputs/labels. The labels are no more in form of a vector, but rather a `DataFrame`
        with two additional features: 'ground_index' and 'monotonicity' (wrt the ground index).
    """
    x, y = x.reset_index(drop=True), y.reset_index(drop=True)
    new_samples = []
    for attribute, (num_augmented, function) in sampling_functions.items():
        if num_augmented > 0:
            samples = np.hstack([x.values] * num_augmented).reshape(-1, x.shape[1])
            samples = pd.DataFrame(samples, columns=x.columns).astype(x.dtypes)
            # handle monotonicities on multiple attribute with tuples (python dict do not allow lists as key)
            if isinstance(attribute, tuple):
                attribute = list(attribute)
            samples[attribute] = function(len(samples))
            samples['ground_index'] = np.repeat(np.arange(len(x)), num_augmented)
            if compute_monotonicities is None:
                samples['monotonicity'] = np.nan
            else:
                monotonicities = []
                for index, group in samples.groupby('ground_index'):
                    monotonicities += list(compute_monotonicities(group.drop(columns='ground_index'), x.iloc[index]))
                samples['monotonicity'] = monotonicities
            samples[y.name] = np.nan
        else:
            samples = x.head(0)
            samples['ground_index'] = []
            samples['monotonicity'] = []
            samples[y.name] = []
        new_samples.append(samples)
    df = pd.concat((x, y), axis=1)
    df['ground_index'] = np.arange(len(df))
    df['monotonicity'] = np.nan if compute_monotonicities is None else 0.0
    aug = pd.concat(new_samples).sort_values('ground_index')
    aug = pd.concat((df, aug)).reset_index(drop=True)
    # there is no way to return integer labels due to the presence of nan values (pd.Int64 type have problems with tf)
    aug = aug.astype({'ground_index': int})
    return aug[x.columns], aug[[y.name, 'ground_index', 'monotonicity']]


def compute_numeric_monotonicities(samples: np.ndarray,
                                   references: np.ndarray,
                                   directions: np.ndarray,
                                   eps: float = 1e-6) -> MonotonicitiesMatrix:
    """Default way of computing monotonicities in case all the features are numeric.

    :param samples:
        The matrix of data points.

    :param references:
        The vector/matrix of reference data point(s).

    :param directions:
        The direction of the monotonicity wrt each feature.

    :param eps:
        The slack value under which a violation is considered to be acceptable.

    :return:
        A NxM matrix where N is the number of samples and M is the number of references, where each cell is filled with
        -1, 0, or 1 depending on the kind of monotonicity between samples[i] and references[j].
    """
    assert samples.ndim <= 2, f"'samples' should have 2 dimensions at most, but it has {samples.ndim}"
    assert references.ndim <= 2, f"'references' should have 2 dimensions at most, but it has {references.ndim}"
    # convert vectors into a matrices
    samples, references = np.atleast_2d(samples), np.atleast_2d(references)
    # increase samples dimension to match references
    samples = np.hstack([samples] * len(references)).reshape((len(samples), len(references), -1))
    # compute differences between samples to get the number of different attributes
    differences = samples - references
    differences[np.abs(differences) < eps] = 0.
    num_differences = np.sign(np.abs(differences)).sum(axis=-1)
    # get whole monotonicity (sum of monotonicity signs) and mask for pairs with just one different attribute
    # directions is either a float or an array that says whether the monotonicity is increasing or decreasing (or null)
    monotonicities = np.sign(directions * differences).sum(axis=-1)
    monotonicities = np.squeeze(monotonicities * (num_differences == 1)).astype('int')
    # if a there is a single sample and a single reference, numpy.sum(axis=-1) will return a zero-dimensional array
    # instead of a scalar, thus it is necessary to manually handle this case
    return np.int32(monotonicities) if monotonicities.ndim == 0 else monotonicities


def get_monotonicities_list(data: pd.DataFrame,
                            kind: str = 'group',
                            label: Optional[str] = None,
                            compute_monotonicities: Callable = None,
                            errors: str = 'raise') -> MonotonicitiesList:
    """Computes the monotonicities list from an augmented dataframe.

    :param data:
        The augmented dataframe.

    :param kind:
        The monotonicity computation modality:

        - 'ground', which computes the monotonicity within each subgroup respectively to the ground index only.
        - 'group', which computes the monotonicity within each subgroup between each pair in the subgroup.
        - 'all', which computes the monotonicity between each pair in the whole dataset (very slow).

    :param label:
        The y label to be dropped, if present.

    :param compute_monotonicities:
        Routine that computes the monotonicities of some data points wrt a reference point, it must be a function
        f(<x>, <y>) -> <monotonicities_matrix> that computes pairwise expected monotonicities between two vectors <x>
        and <y> (with size <n> and <m>, respectively) and stores them in a matrix having shape (<n>, <m>).

    :param errors:
        Pandas argument that sets the behaviour in case a non present column is being dropped.

    :return:
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
            values = group.drop([label, 'ground_index', 'monotonicity'], errors=errors, axis=1)
            his, lis = np.where(compute_monotonicities(values, values) == 1)
            higher_indices.append(group.index.values[his])
            lower_indices.append(group.index.values[lis])
        higher_indices = np.concatenate(higher_indices)
        lower_indices = np.concatenate(lower_indices)
    # all monotonicities: retrieved using the 'compute_monotonicities' function independently from the ground index
    elif kind == 'all':
        values = data.drop([label, 'ground_index', 'monotonicity'], errors=errors, axis=1)
        higher_indices, lower_indices = np.where(compute_monotonicities(values, values) == 1)
    else:
        raise ValueError(f"'{kind}' is not a valid monotonicities kind")
    return [(hi, li) for hi, li in zip(higher_indices, lower_indices)]
