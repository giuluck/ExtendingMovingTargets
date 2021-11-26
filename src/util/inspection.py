"""Dataset Inspection utils."""
import pandas as pd


def set_pandas_options(max_rows: int = 100, max_columns: int = 100, max_colwidth: int = 1000, width: int = 100000):
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.max_colwidth', max_colwidth)
    pd.set_option('display.width', width)
