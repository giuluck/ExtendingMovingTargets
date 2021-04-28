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


def get_augmented_data(x, y, directions=1, num_rand_samples=0, num_aug_samples=5, num_ground_samples=None, seed=0):
    rng = np.random.default_rng(seed=seed)

    # handle directions
    if isinstance(directions, dict):
        directions = [directions.get(c, 0) for c in x.columns]
    elif isinstance(directions, int) or isinstance(directions, float):
        directions = [directions] * x.shape[1]
    directions = np.array(directions)

    # monotonicities routine with directions
    def monotonicities(samples, references):
        return compute_monotonicities(samples, references, directions)

    # handle num samples
    if isinstance(num_aug_samples, dict):
        num_aug_samples = [num_aug_samples.get(c, 0) for c in x.columns]
    elif isinstance(num_aug_samples, int):
        num_aug_samples = [0 if d == 0 else num_aug_samples for d in directions]
    num_aug_samples = np.array(num_aug_samples)

    # handle input samples reduction
    if num_ground_samples is not None:
        x = x.head(num_ground_samples)
        y = y.head(num_ground_samples)

    # add random unsupervised samples to fill the data space
    if num_rand_samples > 0:
        x_rnd = rng.uniform(size=(num_rand_samples, len(x.columns)))
        y_rnd = [np.nan] * num_rand_samples
        x = pd.concat((x, pd.DataFrame(x_rnd, columns=x.columns)), ignore_index=True)
        y = pd.concat((y, pd.Series(y_rnd, name=y.name)), ignore_index=True)

    # augment data
    aug_data, aug_info = augment_data(x, compute_monotonicities=monotonicities, sampling_functions={
        c: (n, lambda s: rng.uniform(0, 1, size=s)) for n, c in zip(num_aug_samples, x.columns)
    })
    x_aug = pd.concat((x, aug_data)).reset_index(drop=True)
    y_aug = pd.concat((y, aug_info)).rename({0: y.name}, axis=1).reset_index(drop=True)
    y_aug = y_aug.fillna({'ground_index': pd.Series(y_aug.index), 'monotonicity': 0}).astype({'ground_index': 'int'})
    return x_aug, y_aug, pd.concat((x_aug, y_aug), axis=1)
