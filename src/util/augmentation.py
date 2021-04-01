import numpy as np
import pandas as pd


def augment_data(df, sampling_functions, compute_monotonicities, n=1):
    new_samples = []
    new_info = []
    for ground_index, x in df.iterrows():
        for attribute, function in sampling_functions.items():
            samples = pd.DataFrame([x] * n)
            # handle monotonicities on multiple attribute with tuples (python dictionaries do not allow lists as key)
            if isinstance(attribute, tuple):
                attribute = list(attribute)
            samples[attribute] = function(n)
            monotonicities = compute_monotonicities(samples.values, x.values.reshape(1, -1)).reshape(-1, )
            new_samples.append(samples.astype(df.dtypes))
            new_info.append(pd.DataFrame(
                data=zip([ground_index] * n, monotonicities),
                columns=['ground_index', 'monotonicity'],
                dtype='int'
            ))
    return pd.concat(new_samples).reset_index(drop=True), pd.concat(new_info).reset_index(drop=True)


def filter_vectors(mask_value, reference_vector, *args):
    args = list(args)
    if mask_value is not None:
        mask = np.isnan(reference_vector) if np.isnan(mask_value) else reference_vector == mask_value
        reference_vector = reference_vector[~mask]
        for i, vector in enumerate(args):
            args[i] = vector[~mask]
    return tuple([reference_vector] + args)
