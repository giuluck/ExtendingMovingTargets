"""Dataset Inspection utils."""
from typing import Optional, Callable

import pandas as pd


def set_pandas_options(min_rows: Optional[int] = 1000,
                       max_rows: Optional[int] = 1000,
                       max_columns: Optional[int] = 1000,
                       max_colwidth: Optional[int] = 1000,
                       width: Optional[int] = 100000,
                       precision: Optional[int] = None,
                       float_format: Optional[Callable] = '{:.2f}'.format):
    for key, value in locals().items():
        if value is not None:
            pd.set_option(f'display.{key}', value)
