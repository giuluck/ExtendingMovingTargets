from typing import Optional, Dict, List, Any


def cartesian_product(fixed_parameters: Optional[Dict[str, List[Any]]] = None, **kwargs):
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


def merge_dictionaries(first_dict: Dict[str, Any], second_dict: Dict[str, Any]):
    first_dict = {} if first_dict is None else first_dict.copy()
    second_dict = {} if second_dict is None else second_dict.copy()
    first_dict.update(second_dict)
    return first_dict
