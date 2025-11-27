from numpy import finfo, float32, pi
from pydantic import ConfigDict, validate_call

#: The smallest positive number such that `1.0 + EPS != 1.0` for 32-bit floating-point values.
EPS: float = float(finfo(float32).eps)

#: Value equal to π = 3.1415926535...
PI: float = pi

#: Value equal to π/2
HALF_PI: float = pi*0.5

#: Value equal to 2*π
TWO_PI: float = pi*2

DEFAULT_CONFIG = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, str_to_lower=True)
VALIDATE_RETURN_CONFIG = DEFAULT_CONFIG | {'validate_return': True}

validate_variables = validate_call(config=VALIDATE_RETURN_CONFIG)
