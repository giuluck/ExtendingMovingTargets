from typing import Any, List, Dict


def cartesian_product(fixed_parameters: Dict[str, Any] = None, **kwargs: List[Any]) -> List[Dict[str, Any]]:
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
