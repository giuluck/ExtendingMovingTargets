import numpy as np
import pandas as pd


def augment_data(df, sampling_functions, compute_monotonicities):
    new_samples = []
    new_info = []
    for ground_index, x in df.iterrows():
        for attribute, (n, function) in sampling_functions.items():
            if n > 0:
                samples = pd.DataFrame([x] * n)
                # handle monotonicities on multiple attribute with tuples (python dict do not allow lists as key)
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


def get_monotonicities_list(data, compute_monotonicities, label, kinds):
    outputs = {}
    for kind in [kinds] if isinstance(kinds, str) else kinds:
        higher_indices, lower_indices = [], []
        if kind == 'ground':
            for idx, rec in data.iterrows():
                monotonicity, ground_index = rec['monotonicity'], int(rec['ground_index'])
                if monotonicity != 0:
                    higher_indices.append(idx if monotonicity > 0 else ground_index)
                    lower_indices.append(ground_index if monotonicity > 0 else idx)
        elif kind == 'group':
            for index, group in data.groupby('ground_index'):
                values = group.drop([label, 'ground_index', 'monotonicity'], axis=1).values
                his, lis = np.where(compute_monotonicities(values, values) == 1)
                higher_indices.append(group.index.values[his])
                lower_indices.append(group.index.values[lis])
            higher_indices = np.concatenate(higher_indices)
            lower_indices = np.concatenate(lower_indices)
        elif kind == 'all':
            values = data.drop([label, 'ground_index', 'monotonicity'], axis=1).values
            higher_indices, lower_indices = np.where(compute_monotonicities(values, values) == 1)
        outputs[kind] = [(hi, li) for hi, li in zip(higher_indices, lower_indices)]
    return outputs[kinds] if isinstance(kinds, str) else outputs


def filter_vectors(mask_value, reference_vector, *args):
    args = list(args)
    if mask_value is not None:
        mask = np.isnan(reference_vector) if np.isnan(mask_value) else reference_vector == mask_value
        reference_vector = reference_vector[~mask]
        for i, vector in enumerate(args):
            args[i] = vector[~mask]
    return tuple([reference_vector] + args)
