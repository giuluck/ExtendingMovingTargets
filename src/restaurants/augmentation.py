import numpy as np
import pandas as pd
from tensorflow.python.keras.utils.np_utils import to_categorical

from src.util.augmentation import augment_data
from src.util.preprocessing import Scaler


def compute_monotonicities(samples, references, eps=1e-5):
    def categorical_monotonicities(diffs):
        mono = 1 * (diffs == 1) - 1 * (diffs == -1)  # DD (2) > D    (1) -> DD - D = 1, D - DD = -1
        mono += 1 * (diffs == -6) - 1 * (diffs == 6)  # DD (2) > DDDD (8) -> DD - DDDD = -6, DDDD - DD = 6
        return mono

    # transpose tensors to get shape (6, ...)
    samples, references = samples.copy().transpose(), references.copy().transpose()
    # store categorical values into single element (D -> 1, DD -> 2, DDD -> 4, DDDD -> 8)
    samples[2] = 2 ** samples[2:6].argmax(axis=0)
    references[2] = 2 ** references[2:6].argmax(axis=0)
    # transpose back with single categorical feature (..., 3) and increase samples dimension to match references
    samples, references = samples[:3].transpose(), references[:3].transpose()
    samples = np.hstack([samples] * len(references)).reshape((-1, len(references), 3))
    # compute differences between samples to get the number of different attributes
    differences = (samples - references).transpose()
    differences[np.abs(differences) < eps] = 0.
    num_differences = np.sign(np.abs(differences)).sum(axis=0)
    # convert categorical differences to monotonicities and get whole monotonicity (sum of monotonicity signs)
    differences[-1] = categorical_monotonicities(differences[-1])
    monotonicities = np.sign(differences).sum(axis=0).transpose()
    # the final monotonicities are masked for pairs with just one different attribute
    monotonicities = monotonicities.astype('int') * (num_differences == 1)
    return monotonicities


def get_augmented_data(x, y, num_augmented_samples=5, num_ground_samples=None):
    if num_ground_samples is not None:
        x = x.head(num_ground_samples)
        y = y.head(num_ground_samples)
    # augment data using a fixed number of samples for each attribute
    n = num_augmented_samples
    aug_data, aug_info = augment_data(x, compute_monotonicities=compute_monotonicities, sampling_functions={
        'avg_rating': (n, lambda s: np.random.uniform(1.0, 5.0, size=s)),
        'num_reviews': (n, lambda s: np.round(np.exp(np.random.uniform(0.0, np.log(200), size=s)))),
        ('D', 'DD', 'DDD', 'DDDD'): (n, lambda s: to_categorical(np.random.randint(4, size=s), num_classes=4))
    })
    # concatenate the data in the desired way
    x_aug = pd.concat((x, aug_data)).reset_index(drop=True)
    y_aug = pd.concat((y, aug_info)).rename({0: 'clicked'}, axis=1).reset_index(drop=True)
    y_aug = y_aug.fillna({'ground_index': pd.Series(y_aug.index), 'monotonicity': 0}).astype({'ground_index': 'int'})
    full_aug = pd.concat((x_aug, y_aug), axis=1)
    aug_scaler = Scaler(methods=dict(avg_rating='std', num_reviews='std')).fit(x_aug)
    return x_aug, y_aug, full_aug, aug_scaler
