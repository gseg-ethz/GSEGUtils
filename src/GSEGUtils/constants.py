from typing import Optional, NamedTuple

from numpy import finfo, float32, pi
from pydantic import ConfigDict, validate_call

EPS = finfo(float32).eps

PI = pi
HALF_PI = pi * 0.5
TWO_PI = pi * 2

DEFAULT_CONFIG = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, str_to_lower=True)
VALIDATE_RETURN_CONFIG = DEFAULT_CONFIG | {'validate_return': True}

validate_variables = validate_call(config=VALIDATE_RETURN_CONFIG)
