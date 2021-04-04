import numpy as np
import pandas as pd

from src.util.augmentation import augment_data


def compute_monotonicities(samples, references, directions, eps=1e-5):
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


def get_augmented_data(x, y, directions: object = 1., n=5, num_ground_samples=None):
    if isinstance(directions, dict):
        directions = [directions.get(c, 0) for c in x.columns]
    elif isinstance(directions, int) or isinstance(directions, float):
        directions = [directions] * x.shape[1]
    directions = np.array(directions)

    def monotonicities(samples, references, eps=1e-5):
        return compute_monotonicities(samples, references, np.array(directions), eps)

    if num_ground_samples is not None:
        x = x.head(num_ground_samples)
        y = y.head(num_ground_samples)
    aug_data, aug_info = augment_data(x, n=n, compute_monotonicities=monotonicities, sampling_functions={
        col: lambda s: np.random.uniform(0.0, 1.0, size=s) for idx, col in enumerate(x.columns) if directions[idx] != 0
    })
    x_aug = pd.concat((x, aug_data)).reset_index(drop=True)
    y_aug = pd.concat((y, aug_info)).rename({0: y.name}, axis=1).reset_index(drop=True)
    y_aug = y_aug.fillna({'ground_index': pd.Series(y_aug.index), 'monotonicity': 0})
    return x_aug, y_aug, pd.concat((x_aug, y_aug), axis=1)
