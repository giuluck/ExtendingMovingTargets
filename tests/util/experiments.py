import random
import numpy as np
import pandas as pd
import tensorflow as tf


def setup(seed=0, max_rows=10000, max_columns=10000, width=10000, max_colwidth=10000, float_format='{:.4f}'):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    pd.options.display.max_rows = max_rows
    pd.options.display.max_columns = max_columns
    pd.options.display.width = width
    pd.options.display.max_colwidth = max_colwidth
    pd.options.display.float_format = float_format.format
