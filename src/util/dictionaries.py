"""Dictionaries utils."""

from typing import Optional, Dict, List


def cartesian_product(fixed_parameters: Optional[Dict] = None, **kwargs: List) -> List[Dict]:
    """Creates a combinatorial list of configurations given a dictionary of fixed parameters and a series of keyword
       arguments representing a list of variable parameters

    Args:
        fixed_parameters: a dictionary of fixed parameters. If None, an empty dictionary is used.
        **kwargs: a dictionary of variable parameters, which may assume each value indicated in the respective list.

    Returns:
        A list of all the possible configuration of parameters, i.e., the cartesian product.
    """
    fixed_parameters = {} if fixed_parameters is None else fixed_parameters
    if len(kwargs) == 0:
        return [fixed_parameters]
    else:
        cart_product = []
        parameter, values = kwargs.popitem()
        for value in values:
            new_parameters = {**fixed_parameters, parameter: value}
            sub_product = cartesian_product(fixed_parameters=new_parameters, **kwargs)
            cart_product.append(sub_product)
        return [parameters for sub_product in cart_product for parameters in sub_product]


def merge_dictionaries(first_dict: Optional[Dict] = None, second_dict: Optional[Dict] = None) -> Dict:
    """Builds a new dictionary by overriding in the first one the values of the second one having the same key.
       The initial dictionaries are left unchanged.

    Args:
        first_dict: the first dictionary.
        second_dict: the second dictionary.

    Returns:
        A new dictionary obtained as {**first_dict, **second_dict}.
    """
    first_dict = {} if first_dict is None else first_dict.copy()
    second_dict = {} if second_dict is None else second_dict.copy()
    first_dict.update(second_dict)
    return first_dict
