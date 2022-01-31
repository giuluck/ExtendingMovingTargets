from typing import Optional, Callable, List, Dict

import pandas as pd


def cartesian_product(fixed_parameters: Optional[Dict] = None, **parameters: List) -> List[Dict]:
    """Creates a combinatorial list of configurations given a dictionary of fixed parameters and a series of keyword
       arguments representing a list of variable parameters."""
    fixed_parameters = {} if fixed_parameters is None else fixed_parameters
    if len(parameters) == 0:
        return [fixed_parameters]
    else:
        cart_product = []
        parameter, values = parameters.popitem()
        for value in values:
            new_parameters = {**fixed_parameters, parameter: value}
            sub_product = cartesian_product(fixed_parameters=new_parameters, **parameters)
            cart_product.append(sub_product)
        return [parameters for sub_product in cart_product for parameters in sub_product]


def set_pandas_options(min_rows: Optional[int] = 1000,
                       max_rows: Optional[int] = 1000,
                       max_columns: Optional[int] = 1000,
                       max_colwidth: Optional[int] = 1000,
                       width: Optional[int] = 100000,
                       precision: Optional[int] = None,
                       float_format: Optional[Callable] = '{:.2f}'.format):
    """Sets a range of pandas options."""
    for key, value in locals().items():
        if value is not None:
            pd.set_option(f'display.{key}', value)
