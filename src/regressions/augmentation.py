import numpy as np
import pandas as pd

from src.util.augmentation import augment_data

# add random unsupervised samples to fill the data space
# if rand_samples > 0:
#     x_rnd = rng.uniform(size=(rand_samples, len(x.columns)))
#     y_rnd = [np.nan] * rand_samples
#     x = pd.concat((x, pd.DataFrame(x_rnd, columns=x.columns)), ignore_index=True)
#     y = pd.concat((y, pd.Series(y_rnd, name=y.name)), ignore_index=True)
