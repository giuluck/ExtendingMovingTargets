import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical


def compute_monotonicity(x1, x2):
    # convert one hot encoding (2:6) to category [1, 2, 3, 4]
    x1 = np.concatenate((x1[:2], [x1[2:6].argmax() + 1]))
    x2 = np.concatenate((x2[:2], [x2[2:6].argmax() + 1]))
    # get monotonic attributes {0: avg_rating, 1: num_reviews, 2: dollar_rating}
    monotonicities = np.argwhere(1 - np.isclose(x1, x2)).reshape(-1)
    # if there is no monotonic attribute or more than one, nothing could be said about the output monotonicity
    if len(monotonicities) != 1:
        return 0
    # if the monotonic attribute is 2 (dollar_rating), we return a value so that: D < DD, DD > D, DDDD < DD, DDDD > DD
    elif monotonicities[0] == 2:
        dollar_rating_monotonicities = {(1, 2): -1, (2, 1): 1, (4, 2): -1, (2, 4): 1}
        return dollar_rating_monotonicities.get((x1[2], x2[2]), 0)
    # if the monotonic attribute is either 0 or 1, we return the sign of the difference
    else:
        attribute = monotonicities[0]
        return int(np.sign(x1[attribute] - x2[attribute]))


def augment_data(df, n=1):
    new_samples = []
    for _, x in df.iterrows():
        new_values = [
            ('avg_rating', np.random.uniform(1.0, 5.0, size=n)),
            ('num_reviews', np.round(np.exp(np.random.uniform(0.0, np.log(200), size=n)))),
            (['D', 'DD', 'DDD', 'DDDD'], to_categorical(np.random.randint(4, size=n), num_classes=4))
        ]
        for attribute, values in new_values:
            samples = pd.DataFrame([x] * n)
            samples[attribute] = values
            samples['monotonicity'] = [compute_monotonicity(s, x) for _, s in samples.iterrows()]
            new_samples.append(samples.astype(df.dtypes))
    return pd.concat(new_samples).reset_index()
