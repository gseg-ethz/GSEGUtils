# GSEGUtils – General utility functions and classes for GSEG research/projects
#
# Copyright (c) 2025–2026 ETH Zurich
# Department of Civil, Environmental and Geomatic Engineering (D-BAUG)
# Institute of Geodesy and Photogrammetry
# Geosensors and Engineering Geodesy
#
# Authors:
#   Nicholas Meyer
#   Jon Allemand
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging

import numpy as np
from numpy import typing as npt

from .constants import HALF_PI, PI, TWO_PI
from .base_types import (
    ArrayT,
    Array_Nx3_Float_T,
    Array_Integer_T,
    Array_Float_T,
    Array_Uint8_T,
    Array_Uint16_T,
    Array_Int8_T,
    Array_Int16_T,
    Array_Int32_T,
    Array_Int64_T,
    Array_NxM_T,
    VectorT
)

logger = logging.getLogger(__name__.split(".")[0])

def validate_spherical_angles(array: Array_Nx3_Float_T) -> Array_Nx3_Float_T:
    """Check spherical coordinates are valid.

    Acceptable ranges:
        Radius: [0, +∞]
        Hz Angle: [-π, +π]
        V Angle: [0, +π]

    Raises errors if not valid. No conversion is done.

    Parameters
    ----------
    array : Array_Nx3_Float_T
        Nx3 array with *rhv* column order (Radius, Horizontal Angle, Vertical Angle)

    Returns
    -------
    Array_Nx3_Float_T

    Raises
    ------
    TypeError
        Non numpy array
    """
    if isinstance(array, np.ndarray):
        array[:, 0] = validate_radius(array[:, 0])
        array[:, 1] = validate_horizontal_angles(array[:, 1])
        array[:, 2] = validate_zenith_angles(array[:, 2])
        return array

    raise TypeError(f"Input values should be an ndarray not : {type(array)}")

def validate_radius(array: Array_Float_T) -> Array_Float_T:
    """ Check radii are non-negative.

    Parameters
    ----------
    array : Array_Float_T
        Input array is independent of the distance unit

    Returns
    -------
    Array_Float_T

    Raises
    ------
    TypeError
        Non numpy array
    ValueError
        Negative values
    """
    if isinstance(array, np.ndarray):
        if np.all(array >= 0):
            return array

        raise ValueError("Radius must be positive")

    raise TypeError(f"Input values should be an ndarray not : {type(array)}")

def validate_azimuth_angles(array: Array_Float_T) -> Array_Float_T:
    """ Check if azimuths in range of [0, 2π]

    Parameters
    ----------
    array : Array_Float_T
        Array of angles in radians

    Returns
    -------
    Array_Float_T

    Raises
    ------
    TypeError
        Non numpy array
    ValueError
        Values outside range [0, 2π]
    """
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
    """ Check if azimuths in range of [-π, +π]

    Parameters
    ----------
    array : Array_Float_T
        Array of angles in radians

    Returns
    -------
    Array_Float_T

    Raises
    ------
    TypeError
        Non numpy array
    ValueError
        Values outside range [-π, +π]
    """
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
    """ Check if zenith angles in range of [0, +π]

    Parameters
    ----------
    array : Array_Float_T
        Array of angles in radians

    Returns
    -------
    Array_Float_T

    Raises
    ------
    TypeError
        Non numpy array
    ValueError
        Values outside range [0, +π]
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input values should be an ndarray not : {type(array)}")

    if 0 <= array.min() and array.max() <= PI:
        return array
    else:
        if -HALF_PI <= array.min() and array.max() <= HALF_PI:
            raise ValueError("Input Angles in [-pi/2, +pi/2] but should be [0, +pi]")
        raise ValueError(f"Zenith angles should be in [0, +pi] not [{array.min()}, {array.max()}]")

def validate_inclination_angles(array: Array_Float_T) -> Array_Float_T:
    """ Check if inclination angles in range of [-π/2, +π/2]

    Parameters
    ----------
    array : Array_Float_T
        Array of angles in radians

    Returns
    -------
    Array_Float_T

    Raises
    ------
    TypeError
        Non numpy array
    ValueError
        Values outside range [-π/2, +π/2]
    """
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
    """ Coerces azimuth angles to be within the range [0, 2π).

    Parameters
    ----------
    array : Array_Float_T
        Array of azimuth angles in radians

    Returns
    -------
    Array_Float_T
    """
    array[array < 0] += TWO_PI
    array[array > TWO_PI] -= TWO_PI
    return array

def coerce_wrapped_horizontal_angles(array: Array_Float_T) -> Array_Float_T:
    """ Coerce horizontal angles to range [-π, π)

    Parameters
    ----------
    array : Array_Float_T
        Array of horizontal angles in radians

    Returns
    -------
    Array_Float_T
    """
    array[array <= -PI] += TWO_PI
    array[array > PI] -= TWO_PI
    return array

def validate_transposed_2d_array(array: Array_NxM_T|VectorT, n: int) -> Array_NxM_T:
    """ Ensure an array is of MxN shape or NxM shape and transpose if necessary.

    Parameters
    ----------
    array : ArrayT
        Input array, either 1D or 2D.
    n : int
        The expected number of columns for the 2D array.

    Returns
    -------
    ArrayT

    Raises
    ------
    ValueError
        Unsupported shape
    """
    if array.ndim == 2:
        # Check if [MxN] or [NxM]
        if array.shape[1] == n:
            return array

        if array.shape[0] == n and array.shape[1] != n:
            return array.T

    elif array.ndim == 1:
        # Check if [1, N]
        if array.shape[0] != n:
            return array.reshape(-1, n)

    raise ValueError(f"Input array must be 2-dimensional of Mx{n} or {n}xM shape. Received: {array.shape}")

def convert_slice_to_integer_range(selection: slice, length: int) -> Array_Integer_T:
    """ Convert a slice object to an array of integer indices.

    Parameters
    ----------
    selection : slice
        Slice object containing start, stop, and step
    length : int
        The length of the target array associated array or sequence

    Returns
    -------
    Array_Integer_T
    """
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
    """ Check if values are within the target range.

    Parameters
    ----------
    value : ArrayT
        Input array
    target_min : float
        Lower inclusive limit
    target_max : float
        Upper inclusive limit

    Raises
    ------
    ValueError
        Values outside of range
    """
    # TODO should be consistent - check usage in PCHandler
    value = np.asarray(value)
    val_min: float | int = value.min()
    val_max: float | int = value.max()

    if (val_min < target_min) and (val_max > target_max):
        raise ValueError(f"Min and max values [{val_min},{val_max}] exceeds bounds [{target_min},{target_max}].")

    elif val_min < target_min:
        raise ValueError(f"Min value {val_min} exceeds lower limit {target_min}.")

    elif val_max > target_max:
        raise ValueError(f"Max value {val_max} exceeds upper limit {target_max}.")

def normalize_min_max(array: ArrayT, lower: float|int, upper: float|int, target_dtype: npt.DtypeLike,
                      v_min: float|int|None=None, v_max: float|int|None=None) -> ArrayT:
    """ Normalize and scale the values in a numpy array to a specified range using min-max scaling.

    The input array is initially normalized using v_min, v_max where array.min() and array.max()
    are the default if not set.

    This is then scaled to the target range bounds of [lower, upper] and converted to the target data type.

    Parameters
    ----------
    array : ArrayT
        Input array
    lower : float or int or np.number
        Target lower bound after normalization
    upper : float or int or np.number
        Target upper bound after normalization
    target_dtype :  npt.DtypeLike
        Output data type
    v_min : float or int, optional
        Minima value used for normalization.
        Defaults to `array.min()` if not provided.
    v_max : float or int, optional
        Maxima value used for normalization.
        Defaults to `array.max()` if not provided.

    Returns
    -------
    ArrayT
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

def linear_map_dtype(array: ArrayT, target_dtype: npt.DtypeLike) -> ArrayT:
    """ Linearly map the array values to the target dtype.

    This function maps the input array values based on the current datatype's minimum
    and maximum supported values to those of the target datatype.

    Examples
    --------
    *np.int8 to np.uint16*
    ::

        >>> input = np.array([-50, 0, 127], dtype=np.int8)
        >>> linear_map_dtype(input, np.uint16)
        array([20046, 32896, 65535], dtype=uint16)


    Parameters
    ----------
    array : ArrayT
        Input array
    target_dtype :  npt.DtypeLike
        Target dtype (64-bit types not supported)

    Returns
    -------
    ArrayT

    Raises
    ------
    ValueError
        Dtype is 64-bit (e.g., np.float64, np.int64, np.uint64).
    TypeError
        Non-floating or integer object type passed
    """

    array = np.asarray(array)

    if target_dtype in (np.float64, np.int64, np.uint64):
        raise ValueError("Target type cannot be 64bit")

    def get_dtype_min_max(dt: npt.DtypeLike) -> tuple[float, float]:
        if np.issubdtype(dt, np.integer):
            return np.iinfo(dt).min, np.iinfo(dt).max
        elif np.issubdtype(dt, np.floating):
            # TODO add check here for values outside of range
            return 0.0, 1.0
        else:
            raise TypeError(f"Invalid dtype detected: {dt}")

    # Types match, exit
    if array.dtype == target_dtype:
        return array

    # Get the corresponding min and max from the type info
    origin_min, origin_max = get_dtype_min_max(array.dtype)
    target_min, target_max = get_dtype_min_max(target_dtype)

    return normalize_min_max(array=array,
                             lower=target_min,
                             upper=target_max,
                             target_dtype=target_dtype,
                             v_min=origin_min,
                             v_max=origin_max)

def normalize_self(array: ArrayT) -> ArrayT:
    """Normalizes an input array to the limits expected by the array's dtype.

    For floating point values, this is `[0, 1]` and for integer values, it is the min and max defined by `np.iinfo`.

    Parameters
    ----------
    array : ArrayT
        The input array to be normalized

    Returns
    -------
    ArrayT
    """
    if np.dtype(array.dtype).kind not in ["u", "i"]:
        logger.debug(f"Scalar field is floating. Converting to [0.0, 1.0].")
        lower, upper = 0, 1
    else:
        lower, upper = np.iinfo(array.dtype).min, np.iinfo(array.dtype).max

    return normalize_min_max(array, lower, upper, array.dtype)

def _normalize_base(array: ArrayT, target_dtype: npt.DtypeLike) -> ArrayT:
    """Helper function for normalizing input array values to target_dtype limits.

    First, normalizes to [0,1] using the array.min() and array.max() values.
    Then scales to dtype `np.iinfo(target_dtype)` min and mix.

    Parameters
    ----------
    array : ArrayT
        The input array to be normalized
    target_dtype : npt.DtypeLike
        Tar

    Returns
    -------
    ArrayT
    """
    array = np.asarray(array)

    if array.dtype != target_dtype:
        if np.issubdtype(target_dtype, np.floating):
            return normalize_min_max(array, 0, 1, target_dtype)

        if 0 <= array.min() <= array.max() <= 1:
            return normalize_min_max(array, np.iinfo(target_dtype).min, np.iinfo(target_dtype).max, target_dtype, 0, 1)

        elif -1 <= array.min() <= array.max() <= 1:
            return normalize_min_max(array, np.iinfo(target_dtype).min, np.iinfo(target_dtype).max, target_dtype, -1, 1)

        return normalize_min_max(array, np.iinfo(target_dtype).min, np.iinfo(target_dtype).max, target_dtype)
    return array

def normalize_uint8(array: ArrayT) -> Array_Uint8_T:
    """Normalize to UInt8

    Parameters
    ----------
    array : ArrayT

    Returns
    -------
    Array_Uint8_T
    """
    return _normalize_base(array, np.uint8)

def normalize_uint16(array: ArrayT) -> Array_Uint16_T:
    """Normalize to UInt16

    Parameters
    ----------
    array : ArrayT

    Returns
    -------
    Array_Uint16_T
    """
    return _normalize_base(array, np.uint16)

def normalize_int8(array: ArrayT) -> Array_Int8_T:
    """Normalize to Int8

    Parameters
    ----------
    array : ArrayT

    Returns
    -------
    Array_Int8_T
    """
    return _normalize_base(array, np.int8)

def normalize_int16(array: ArrayT) -> Array_Int16_T:
    """Normalize to Int16

    Parameters
    ----------
    array : ArrayT

    Returns
    -------
    Array_Int16_T
    """
    return _normalize_base(array, np.int16)

def normalize_int32(array: ArrayT) -> Array_Int32_T:
    """Normalize to Int32

    Parameters
    ----------
    array : ArrayT

    Returns
    -------
    Array_Int32_T
    """
    return _normalize_base(array, np.int32)

def normalize_int64(array: ArrayT) -> Array_Int64_T:
    """Normalize to Int64

    Parameters
    ----------
    array : ArrayT

    Returns
    -------
    Array_Int64_T
    """
    return _normalize_base(array, np.int64)

# TODO normalize functions should be in utils