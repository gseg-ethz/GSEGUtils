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

"""Input-validation helpers for spherical / Cartesian arrays and unsigned-integer normalization.

Provides validators for spherical-coordinate arrays, axis/range checks for angle
columns, transposition helpers for ``Nx2`` / ``Nx3`` arrays, slice-to-integer
conversion, and ``normalize_uint*`` saturation routines.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy import typing as npt

from .base_types import (
    Array_Float_T,
    Array_Int8_T,
    Array_Int16_T,
    Array_Int32_T,
    Array_Int64_T,
    Array_Integer_T,
    Array_Nx3_Float_T,
    Array_NxM_T,
    Array_Uint8_T,
    Array_Uint16_T,
    ArrayT,
    VectorT,
)
from .constants import HALF_PI, PI, TWO_PI

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
    """Check radii are non-negative.

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
    """Check that azimuth angles lie in ``[0, 2π]``.

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
    """Check that horizontal angles lie in ``[-π, +π]``.

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
    """Check that zenith angles lie in ``[0, +π]``.

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
    """Check that inclination angles lie in ``[-π/2, +π/2]``.

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
    """Coerces azimuth angles to be within the range [0, 2π).

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
    """Coerce horizontal angles to the half-open range ``[-π, π)``.

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


def validate_transposed_2d_array(array: Array_NxM_T | VectorT, n: int) -> Array_NxM_T:
    """Ensure an array is of MxN shape or NxM shape and transpose if necessary.

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
    """Convert a slice object to an array of integer indices.

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
    """Validate that all values in ``value`` lie within ``[target_min, target_max]``.

    Contract independent of any specific caller library. Inclusive bounds on both
    sides; the combined out-of-range case (both ``min < target_min`` AND
    ``max > target_max``) raises a single combined-message ``ValueError`` distinct
    from the single-side branches. This is intentional, not a redundant branch.

    Parameters
    ----------
    value : ArrayT
        Input array. Coerced via ``np.asarray`` before reduction.
    target_min : float
        Inclusive lower bound.
    target_max : float
        Inclusive upper bound.

    Raises
    ------
    ValueError
        If any value lies outside ``[target_min, target_max]``. Three branches:
        single-side-low, single-side-high, or combined dual-out-of-range.
    """
    value = np.asarray(value)
    val_min: float | int = value.min()
    val_max: float | int = value.max()

    if (val_min < target_min) and (val_max > target_max):
        raise ValueError(f"Min and max values [{val_min},{val_max}] exceeds bounds [{target_min},{target_max}].")

    elif val_min < target_min:
        raise ValueError(f"Min value {val_min} exceeds lower limit {target_min}.")

    elif val_max > target_max:
        raise ValueError(f"Max value {val_max} exceeds upper limit {target_max}.")


def normalize_min_max(
    array: ArrayT,
    lower: float | int,
    upper: float | int,
    target_dtype: npt.DtypeLike,
    v_min: float | int | None = None,
    v_max: float | int | None = None,
) -> ArrayT:
    """Normalize and scale the values in a numpy array to a specified range using min-max scaling.

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

    if (
        not np.issubdtype(array.dtype, np.floating)
        and not np.issubdtype(array.dtype, np.integer)
        and not np.issubdtype(array.dtype, np.bool)
    ):
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


def linear_map_dtype(
    array: ArrayT,
    target_dtype: npt.DtypeLike,
    *,
    source_range: tuple[float, float] = (0.0, 1.0),
) -> ArrayT:
    """Linearly map the array values to the target dtype.

    This function maps the input array values based on the current datatype's minimum
    and maximum supported values to those of the target datatype.

    Float-input path: clip to ``source_range`` then linearly map to the
    target dtype's :func:`numpy.iinfo` range (clip-and-saturate, Phase 4
    D-12 / D-16). NaN / +Inf / -Inf input raises ``ValueError`` (D-14).
    Integer-input path: passes through ``normalize_min_max`` against the
    source dtype's :func:`numpy.iinfo`; ``source_range`` is silently
    ignored when ``array.dtype.kind in ('u', 'i')`` (D-15).

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
    source_range : tuple[float, float], keyword-only, default (0.0, 1.0)
        Inclusive range of the float input that maps onto the target dtype's
        full ``iinfo`` extent. Values outside the range are clipped before
        scaling (D-12). Silently ignored for integer input (D-15).

    Returns
    -------
    ArrayT

    Raises
    ------
    ValueError
        Dtype is 64-bit (e.g., np.float64, np.int64, np.uint64); or the
        float input contains NaN / +Inf / -Inf (D-14); or ``source_range``
        bounds are non-finite or not strictly increasing.
    TypeError
        Non-floating or integer object type passed

    Notes
    -----
    Phase 4 D-12 / D-16: the clip-and-saturate policy replaces the prior
    pass-through behaviour for unbounded floats. Callers that relied on the
    default ``[0, 1]`` source range continue to work unchanged; callers
    that fed signed-float input ``[-1, 1]`` must pass
    ``source_range=(-1.0, 1.0)`` explicitly.
    """
    lower, upper = source_range
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        raise ValueError(f"source_range must be finite and lower<upper; got ({lower}, {upper})")

    array = np.asarray(array)

    if target_dtype in (np.float64, np.int64, np.uint64):
        raise ValueError("Target type cannot be 64bit")

    # Types match, exit
    if array.dtype == target_dtype:
        return array

    # Compute source-domain (v_min, v_max) per dtype family.
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"linear_map_dtype: input contains NaN/Inf (target_dtype={np.dtype(target_dtype).name})")
        array = np.clip(array, lower, upper)
        v_min, v_max = float(lower), float(upper)
    elif np.issubdtype(array.dtype, np.integer):
        # D-15: integer-input path ignores source_range.
        v_min = float(np.iinfo(array.dtype).min)
        v_max = float(np.iinfo(array.dtype).max)
    else:
        raise TypeError(f"Invalid source dtype detected: {array.dtype}")

    # Dispatch on target dtype family.
    if np.issubdtype(target_dtype, np.floating):
        return normalize_min_max(array, 0.0, 1.0, target_dtype, v_min, v_max)
    if np.issubdtype(target_dtype, np.integer):
        return normalize_min_max(
            array,
            np.iinfo(target_dtype).min,
            np.iinfo(target_dtype).max,
            target_dtype,
            v_min,
            v_max,
        )
    raise TypeError(f"Invalid target dtype detected: {target_dtype}")


def normalize_self(array: ArrayT) -> ArrayT:
    """Normalize an input array to the limits expected by its own dtype.

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
        logger.debug("Scalar field is floating. Converting to [0.0, 1.0].")
        lower, upper = 0, 1
    else:
        lower, upper = np.iinfo(array.dtype).min, np.iinfo(array.dtype).max

    return normalize_min_max(array, lower, upper, array.dtype)


def _normalize_base(
    array: ArrayT,
    target_dtype: npt.DtypeLike,
    *,
    source_range: tuple[float, float] = (0.0, 1.0),
) -> ArrayT:
    """Normalize ``array`` to the limits of ``target_dtype``.

    Float-input path: clip to ``source_range``, then linearly scale to the
    target dtype's range (``[0.0, 1.0]`` for float targets,
    :func:`numpy.iinfo` extremes for integer targets). NaN / +Inf / -Inf
    raises ``ValueError``.

    Integer-input path: passes through :func:`normalize_min_max` against
    the target dtype's extremes. ``source_range`` is silently ignored for
    integer inputs.

    Parameters
    ----------
    array : ArrayT
        The input array to be normalized.
    target_dtype : npt.DtypeLike
        The target numpy dtype. Float and integer dtypes are supported.
    source_range : tuple[float, float], keyword-only, default (0.0, 1.0)
        Inclusive ``(lower, upper)`` interval of the float input that maps
        onto the target dtype's full range. Values outside ``source_range``
        are clipped before scaling (Phase 4 D-12). The bounds must be
        finite and strictly increasing — misuse raises ``ValueError``.

    Returns
    -------
    ArrayT

    Raises
    ------
    ValueError
        If ``source_range`` bounds are non-finite or not strictly
        increasing, or if a float input contains NaN / +Inf / -Inf
        (Phase 4 D-14).

    Notes
    -----
    Phase 4 D-12 / D-16: the clip-and-saturate policy replaces the prior
    auto-detect branching on ``[0, 1]`` / ``[-1, 1]`` / min-max-rescale.
    Callers that relied on the auto-detect behaviour (e.g. signed-float
    input ``[-1, 1]``) must pass ``source_range=(-1.0, 1.0)`` explicitly.
    Integer-input semantics are unchanged; ``source_range`` is silently
    ignored when ``array.dtype.kind in ('u', 'i')`` (Phase 4 D-15).
    """
    lower, upper = source_range
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        raise ValueError(f"source_range must be finite and lower<upper; got ({lower}, {upper})")

    array = np.asarray(array)

    if array.dtype != target_dtype:
        if np.issubdtype(array.dtype, np.floating):
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"_normalize_base: input contains NaN/Inf (target_dtype={np.dtype(target_dtype).name})"
                )
            array = np.clip(array, lower, upper)
            if np.issubdtype(target_dtype, np.floating):
                # float -> float: linear remap [lower, upper] -> [0, 1]
                return normalize_min_max(array, 0.0, 1.0, target_dtype, lower, upper)
            return normalize_min_max(
                array,
                np.iinfo(target_dtype).min,
                np.iinfo(target_dtype).max,
                target_dtype,
                lower,
                upper,
            )

        # Integer-input path: unchanged from prior implementation
        # (source_range silently ignored per D-15)
        if np.issubdtype(target_dtype, np.floating):
            return normalize_min_max(array, 0.0, 1.0, target_dtype)
        return normalize_min_max(
            array,
            np.iinfo(target_dtype).min,
            np.iinfo(target_dtype).max,
            target_dtype,
        )
    return array


def normalize_uint8(array: ArrayT, *, source_range: tuple[float, float] = (0.0, 1.0)) -> Array_Uint8_T:
    """Normalize ``array`` into the ``uint8`` range.

    Float-input path: clip to ``source_range`` then linearly scale to
    ``[0, 255]`` (clip-and-saturate, Phase 4 D-12). NaN / +Inf / -Inf
    raises ``ValueError`` (D-14). Integer-input path: ``source_range`` is
    silently ignored (D-15).

    Parameters
    ----------
    array : ArrayT
        Input array (float or integer dtype).
    source_range : tuple[float, float], keyword-only, default (0.0, 1.0)
        Inclusive interval of the float input that maps onto ``[0, 255]``.
        See :func:`_normalize_base` for full semantics.

    Returns
    -------
    Array_Uint8_T
    """
    return _normalize_base(array, np.uint8, source_range=source_range)


def normalize_uint16(array: ArrayT, *, source_range: tuple[float, float] = (0.0, 1.0)) -> Array_Uint16_T:
    """Normalize ``array`` into the ``uint16`` range.

    Float-input path: clip to ``source_range`` then linearly scale to
    ``[0, 65535]`` (clip-and-saturate, Phase 4 D-12). NaN / +Inf / -Inf
    raises ``ValueError`` (D-14). Integer-input path: ``source_range`` is
    silently ignored (D-15).

    Parameters
    ----------
    array : ArrayT
        Input array (float or integer dtype).
    source_range : tuple[float, float], keyword-only, default (0.0, 1.0)
        Inclusive interval of the float input that maps onto ``[0, 65535]``.
        See :func:`_normalize_base` for full semantics.

    Returns
    -------
    Array_Uint16_T
    """
    return _normalize_base(array, np.uint16, source_range=source_range)


def normalize_int8(array: ArrayT) -> Array_Int8_T:
    """Normalize ``array`` into the ``int8`` range.

    Parameters
    ----------
    array : ArrayT

    Returns
    -------
    Array_Int8_T
    """
    return _normalize_base(array, np.int8)


def normalize_int16(array: ArrayT) -> Array_Int16_T:
    """Normalize ``array`` into the ``int16`` range.

    Parameters
    ----------
    array : ArrayT

    Returns
    -------
    Array_Int16_T
    """
    return _normalize_base(array, np.int16)


def normalize_int32(array: ArrayT) -> Array_Int32_T:
    """Normalize ``array`` into the ``int32`` range.

    Parameters
    ----------
    array : ArrayT

    Returns
    -------
    Array_Int32_T
    """
    return _normalize_base(array, np.int32)


def normalize_int64(array: ArrayT) -> Array_Int64_T:
    """Normalize ``array`` into the ``int64`` range.

    Parameters
    ----------
    array : ArrayT

    Returns
    -------
    Array_Int64_T
    """
    return _normalize_base(array, np.int64)


# TODO normalize functions should be in utils
