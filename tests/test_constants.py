from typing import Callable

import numpy as np

from GSEGUtils.constants import (
    DEFAULT_CONFIG,
    EPS,
    HALF_PI,
    PI,
    TWO_PI,
    VALIDATE_RETURN_CONFIG,
    validate_variables,
)


def test_eps():
    assert np.finfo(np.float32).eps == EPS


def test_PI_values():
    assert PI == np.pi
    assert HALF_PI == np.pi / 2
    assert TWO_PI == np.pi * 2


def test_default_pydantic_config():
    assert DEFAULT_CONFIG.get("arbitrary_types_allowed", False)
    assert DEFAULT_CONFIG.get("validate_assignment", True)
    assert DEFAULT_CONFIG.get("str_to_lower", True)
    assert VALIDATE_RETURN_CONFIG.get("validate_return", True)
    assert isinstance(validate_variables, Callable)
