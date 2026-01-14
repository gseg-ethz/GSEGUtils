from typing import Callable

import numpy as np


def compare_ndarray_nested_dicts(
    a: dict, b: dict, np_comp_method: Callable = np.array_equal, **kwargs
) -> bool:
    if a.keys() != b.keys():
        return False

    for key, value in a.items():
        if isinstance(value, np.ndarray):
            if not np_comp_method(value, b[key], **kwargs):
                return False
        else:
            if value != b[key]:
                return False

    return True
