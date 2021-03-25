import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical


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
    samples = np.hstack([samples] * len(references)).reshape((len(references), -1, 3))
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


def augment_data(df, n=1):
    new_samples = []
    new_info = []
    for ground_index, x in df.iterrows():
        new_values = [
            ('avg_rating', np.random.uniform(1.0, 5.0, size=n)),
            ('num_reviews', np.round(np.exp(np.random.uniform(0.0, np.log(200), size=n)))),
            (['D', 'DD', 'DDD', 'DDDD'], to_categorical(np.random.randint(4, size=n), num_classes=4))
        ]
        for attribute, values in new_values:
            samples = pd.DataFrame([x] * n)
            samples[attribute] = values
            monotonicities = compute_monotonicities(samples.values, x.values.reshape(1, -1)).reshape(-1, )
            new_samples.append(samples.astype(df.dtypes))
            new_info.append(pd.DataFrame(
                data=zip([ground_index] * n, monotonicities),
                columns=['ground_index', 'monotonicity'],
                dtype='int'
            ))
    return pd.concat(new_samples).reset_index(drop=True), pd.concat(new_info).reset_index(drop=True)
