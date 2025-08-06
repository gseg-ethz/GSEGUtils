from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy import typing as npt

from .constants import HALF_PI, PI, TWO_PI
from .base_types import (
    ArrayT,
    Array_Integer_T, Array_Float_T,
    Array_Uint8_T, Array_Uint16_T,
    Array_Int8_T, Array_Int16_T, Array_Int32_T, Array_Int64_T,
)

logger = logging.getLogger(__name__.split(".")[0])

def validate_spherical_angles(array: Array_Float_T) -> Array_Float_T:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input values should be an ndarray not : {type(array)}")

    array[:, 0] = validate_radius(array[:, 0])
    array[:, 1] = validate_horizontal_angles(array[:, 1])
    array[:, 2] = validate_zenith_angles(array[:, 2])
    return array

def validate_radius(array: Array_Float_T) -> Array_Float_T:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input values should be an ndarray not : {type(array)}")

    if np.any(array < 0):
        raise ValueError("Radius must be positive")
    return array

def validate_azimuth_angles(array: Array_Float_T) -> Array_Float_T:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input values should be an ndarray not : {type(array)}")

    if 0 <= array.min() and array.max() <= TWO_PI:
        return array
    else:
        if -PI <= array.min() and array.max() <= PI:
            arr_min, arr_max = -PI, PI
        else:
            arr_min, arr_max = array.min(), array.max()

        raise ValueError(f"Azimuths must be between [0, 2*pi] not [{arr_min}, {arr_max}]")

def validate_horizontal_angles(array: Array_Float_T) -> Array_Float_T:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input values should be an ndarray not : {type(array)}")

    if -PI <= array.min() and array.max() <= PI:
        return array
    else:
        if 0 <= array.min() and array.max() <= PI * 2:
            arr_min, arr_max = -PI, PI
        else:
            arr_min, arr_max = array.min(), array.max()

        raise ValueError(f"Horizontal angles must be between [-pi, +pi] not [{arr_min}, {arr_max}]")

def validate_zenith_angles(array: Array_Float_T) -> Array_Float_T:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input values should be an ndarray not : {type(array)}")

    if 0 <= array.min() and array.max() <= PI:
        return array
    else:
        if -HALF_PI <= array.min() and array.max() <= HALF_PI:
            raise ValueError("Input Angles in [-pi/2, +pi/2] but should be [0, +pi]")
        raise ValueError(f"Zenith angles should be in [0, +pi] not [{array.min()}, {array.max()}]")

def validate_inclination_angles(array: Array_Float_T) -> Array_Float_T:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input values should be an ndarray not : {type(array)}")

    if -HALF_PI <= array.min() and array.max() <= HALF_PI:
        return array
    else:
        if 0 <= array.min() and array.max() <= PI:
            array_min, array_max = 0, PI
        else:
            array_min, array_max = array.min(), array.max()
        raise ValueError(f"Inclination angles should be between [-pi/2, +pi/2] not [{array_min}, {array_max}]")

def coerce_wrapped_azimuth_angles(array: Array_Float_T) -> Array_Float_T:
    array[array < 0] += TWO_PI
    array[array > TWO_PI] -= TWO_PI
    return array

def coerce_wrapped_horizontal_angles(array: Array_Float_T) -> Array_Float_T:
    array[array <= -PI] += TWO_PI
    array[array > PI] -= TWO_PI
    return array

def validate_transposed_2d_array(array: ArrayT, cols: int) -> ArrayT:
    """
    Validates the transpose of 2D arrays with known fixed cols
    :param array:
    :param cols:
    :return:
    """
    if array.ndim == 2:
        if array.shape[1] == cols:
            return array

        if array.shape[0] == cols and array.shape[1] != cols:
            return array.T

    elif array.ndim == 1:
        if array.shape[0] != cols:
            return array.reshape(-1, cols)

    raise ValueError(f"Input array must be 2-dimensional of Nx{cols} or {cols}xN shape. Received: {array.shape}")

def convert_slice_to_integer_range(selection: slice, length: int) -> Array_Integer_T:
    start = selection.start
    stop = selection.stop
    step = selection.step

    # Default
    if step is None:
        step = 1

    if start is None:
        # If `step` is positive, start at 0. if `step` is negative, start from the end of the array
        start = 0 if step > 0 else length - 1
    elif start < 0:
        # Convert negative addresses to positive address
        start += length
    else:
        pass

    if stop is None:
        # Set stop point to include endpoint values if None is set
        stop = length if step > 0 else -1
    elif stop < 0:
        # Convert negative index to positive index
        stop += length

    # Convert slice objects to a numpy integer array
    return np.arange(start=start, stop=stop, step=step)

def validate_in_range(value: ArrayT, target_min: float, target_max: float) -> None:
    value = np.asarray(value)
    val_min: float | int = value.min()
    val_max: float | int = value.max()

    if (val_min < target_min) and (val_max > target_max):
        raise ValueError(f"Min and max values [{val_min},{val_max}] exceeds bounds [{target_min},{target_max}].")

    elif val_min < target_min:
        raise ValueError(f"Min value {val_min} exceeds lower limit {target_min}.")

    elif val_max > target_max:
        raise ValueError(f"Max value {val_max} exceeds upper limit {target_max}.")

def normalize_min_max(array: ArrayT,
                      lower: float|int|np.number,
                      upper: float|int|np.number,
                      target_dtype: npt.DTypeLike,
                      v_min: Optional[float|int] = None,
                      v_max: Optional[float|int] = None) -> ArrayT:
    """
    Function that can perform normalisation of an input data type.

    Allows input of v_min and v_max to override the min and max of the input array for either clipping or
    linear mapping between datatypes.

    Parameters
    ----------
    array : ArrayT
    lower: float|int|np.number
    upper: float|int|np.number
    target_dtype: npt.DTypeLike
        Datatype of numpy array to return.
    v_min: Optional[float|int] = None
        lower bound value of input array to scale to (defaults to array.min())
    v_max: Optional[float|int] = None
        upper bound value of input array to scale to (defaults to array.max())

    Returns
    -------

    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input array must be a numpy array not {type(array)}")

    if (not np.issubdtype(array.dtype, np.floating) and
            not np.issubdtype(array.dtype, np.integer) and
            not np.issubdtype(array.dtype, np.bool)):
        raise TypeError(f"Cannot convert numpy array of type {array.dtype}")

    array = array.astype(np.float64)

    if v_min is None:
        v_min = array.min()

    if v_max is None:
        v_max = array.max()

    if v_max <= v_min:
        raise ValueError(f"v_max ({v_max}) must be greater than v_min ({v_min})")

    array = (array - v_min) / (v_max - v_min)
    array = np.add(array * (upper - lower), lower)
    return np.clip(array, lower, upper).astype(target_dtype)

def linear_map_dtype(array: ArrayT, target_dtype: npt.DTypeLike) -> ArrayT:
    """
    Performs a linear mapping between datatypes assuming the min and max value supported by each dtype.
    Parameters
    ----------
    array: ArrayT
    target_dtype: npt.DTypeLike

    Returns
    -------

    """

    array = np.asarray(array)

    if target_dtype in (np.float64, np.int64, np.uint64):
        raise ValueError("Target type cannot be 64bit")

    def get_dtype_min_max(dt: np.dtype) -> tuple[float, float]:
        if np.issubdtype(dt, np.integer):
            return np.iinfo(dt).min, np.iinfo(dt).max
        elif np.issubdtype(dt, np.floating):
            return 0.0, 1.0
        else:
            raise TypeError(f"Invalid dtype detected: {dt}")

    # Types match, exit
    if array.dtype == target_dtype:
        return array

    # Get the corresponding min and max from the type info
    origin_min, origin_max = get_dtype_min_max(array.dtype)
    target_min, target_max = get_dtype_min_max(np.dtype(target_dtype))

    return normalize_min_max(array=array,
                             lower=target_min,
                             upper=target_max,
                             target_dtype=target_dtype,
                             v_min=origin_min,
                             v_max=origin_max)

def normalize_self(array: ArrayT) -> ArrayT:
    """
    Returns the input array normalised to the datatype min and max values for integer types and [0, 1] for floating

    e.g points [2, 10] for np.uint8 would be linear mapped to [0, 255]

    """
    if np.dtype(array.dtype).kind not in ["u", "i"]:
        logger.debug(f"Scalar field is floating. Converting to [0.0, 1.0].")
        lower, upper = 0, 1
    else:
        lower, upper = np.iinfo(array.dtype).min, np.iinfo(array.dtype).max

    return normalize_min_max(array, lower, upper, array.dtype)

def _normalize_base(array: ArrayT, dtype: npt.DTypeLike) -> ArrayT:
    """Base normalisation function that will normalise any datatype to a floating point, [0,1] range"""
    array = np.asarray(array)

    if array.dtype != dtype:
        if np.issubdtype(dtype, np.floating):
            return normalize_min_max(array, 0, 1, dtype)

        if 0 <= array.min() <= array.max() <= 1:
            return normalize_min_max(array, np.iinfo(dtype).min, np.iinfo(dtype).max, dtype, 0, 1)

        return normalize_min_max(array, np.iinfo(dtype).min, np.iinfo(dtype).max, dtype)
    return array

def normalize_uint8(array: ArrayT) -> Array_Uint8_T:
    return _normalize_base(array, np.uint8)

def normalize_uint16(array: ArrayT) -> Array_Uint16_T:
    return _normalize_base(array, np.uint16)

def normalize_int8(array: ArrayT) -> Array_Int8_T:
    return _normalize_base(array, np.int8)

def normalize_int16(array: ArrayT) -> Array_Int16_T:
    return _normalize_base(array, np.int16)

def normalize_int32(array: ArrayT) -> Array_Int32_T:
    return _normalize_base(array, np.int32)

def normalize_int64(array: ArrayT) -> Array_Int64_T:
    return _normalize_base(array, np.int64)
