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

"""Angle-conversion helpers and fast-path NumPy utilities.

Provides :class:`AngleUnit` (rad / deg / gon enum), :func:`convert_angles` for
pair-wise unit conversion, the underscore-prefixed in-place conversion aliases
(scheduled for promotion to public names in Plan 01-04), and
:func:`unique_rows_fast` (a faster alternative to ``numpy.unique(..., axis=0)``
for integer row de-duplication).
"""

import logging
import warnings
from enum import StrEnum
from typing import Optional, cast

import numpy as np
import numpy.typing as npt

from .base_types import Array_Float_T, Array_Int32_T, ArrayT

__all__ = [
    "AngleUnit",
    "convert_angles",
    "rad2deg",
    "rad2gon",
    "deg2rad",
    "deg2gon",
    "gon2rad",
    "gon2deg",
    "unique_rows_fast",
]

logger = logging.getLogger(__name__.split(".")[0])


class AngleUnit(StrEnum):
    """Enumerator for angular units.

    * AngleUnit.RAD = 'rad'
    * AngleUnit.DEGREE = 'deg'
    * AngleUnit.GON = 'gon'

    """

    RAD = "rad"
    DEGREE = "deg"
    GON = "gon"


def convert_angles(  # noqa: C901  # Pair-wise unit conversion dispatch — branching tracks the 3x3 unit matrix; refactor deferred to Phase 6.
    values: Array_Float_T,
    source_unit: AngleUnit,
    target_unit: AngleUnit,
    out: Optional[Array_Float_T] = None,
) -> Array_Float_T | None:
    """Convert an array of angles from one unit to another.

    Parameters
    ----------
    values : Array_Float_T
        Array of angles to convert.
    source_unit : AngleUnit
        Unit of the input angles
    target_unit : AngleUnit
        Unit to convert the angles to.
    out : Optional[Array_Float_T], default=None
        Optional output array to store the results. If provided, it must be the same shape as `values`.

    Returns
    -------
    Array_Float_T

    Notes
    -----
    - If `source_unit` and `target_unit` are the same, the function returns a copy of the input `values` unless
      `out` is provided, in which case it writes the result to `out`.
    - Supported conversions:
      - Radians ↔ Degrees
      - Radians ↔ Gradians
      - Degrees ↔ Gradians

    Examples
    --------
    Convert an array of angles from degrees to radians

    ::

        >>> import numpy as np
        >>> from pchandler.util import convert_angles, AngleUnit
        >>> angles_deg = np.array([0, 90, 180, 360])
        >>> convert_angles(angles_deg, AngleUnit.DEGREE, AngleUnit.RAD)
        array([0.        , 1.57079633, 3.14159265, 6.28318531])

    Convert angles from radians to gradians
    ::

        >>> angles_rad = np.array([0, np.pi/2, np.pi, 2*np.pi])
        >>> convert_angles(angles_rad, AngleUnit.RAD, AngleUnit.GON)
        array([ 0., 100., 200., 400.])

    """
    if source_unit not in AngleUnit:
        raise ValueError(f"Invalid source unit: {source_unit}")

    if target_unit not in AngleUnit:
        raise ValueError(f"Invalid target unit: {target_unit}")

    if out is not None:
        out = cast(npt.NDArray[np.floating], out)
        if not isinstance(out, np.ndarray):
            logger.warning("Input values are not an ndarray, returning None and not assigning converted values")
            return None

    if source_unit == target_unit:
        if out is None:
            return values.copy()
        else:
            if isinstance(values, np.ndarray):
                out[...] = values
            return None

    elif source_unit == AngleUnit.RAD:
        if target_unit == AngleUnit.DEGREE:
            return rad2deg(values) if out is None else rad2deg(values, out=out)
        else:
            return rad2gon(values, out=out)

    elif source_unit == AngleUnit.DEGREE:
        if target_unit == AngleUnit.RAD:
            return deg2rad(values, out=out)
        else:
            return deg2gon(values, out=out)

    else:
        if target_unit == AngleUnit.RAD:
            return gon2rad(values, out=out)
        else:
            return gon2deg(values, out=out)


def rad2deg(values: Array_Float_T | float, out: Optional[Array_Float_T] = None) -> Array_Float_T | float | None:
    """Convert radians to degrees.

    Parameters
    ----------
    values : Array_Float_T or float
        Input angle(s) in radians.
    out : Array_Float_T, optional
        Output array for in-place writes. Default = None.

    Returns
    -------
    Array_Float_T, float, or None
        Converted angle(s) in degrees, or ``None`` when ``out`` is provided.
    """
    return np.rad2deg(values, out=out)


def rad2gon(values: Array_Float_T | float, out: Optional[Array_Float_T] = None) -> Array_Float_T | float | None:
    """Convert radians to gradians (gon).

    Parameters
    ----------
    values : Array_Float_T or float
        Input angle(s) in radians.
    out : Array_Float_T, optional
        Output array for in-place writes. Default = None.

    Returns
    -------
    Array_Float_T, float, or None
        Converted angle(s) in gradians, or ``None`` when ``out`` is provided.
    """
    return np.multiply(values, 200 / np.pi, out=out)


def deg2rad(values: Array_Float_T | float, out: Optional[Array_Float_T] = None) -> Array_Float_T | float | None:
    """Convert degrees to radians.

    Parameters
    ----------
    values : Array_Float_T or float
        Input angle(s) in degrees.
    out : Array_Float_T, optional
        Output array for in-place writes. Default = None.

    Returns
    -------
    Array_Float_T, float, or None
        Converted angle(s) in radians, or ``None`` when ``out`` is provided.
    """
    return np.deg2rad(values, out=out)


def deg2gon(values: Array_Float_T | float, out: Optional[Array_Float_T] = None) -> Array_Float_T | float | None:
    """Convert degrees to gradians (gon).

    Parameters
    ----------
    values : Array_Float_T or float
        Input angle(s) in degrees.
    out : Array_Float_T, optional
        Output array for in-place writes. Default = None.

    Returns
    -------
    Array_Float_T, float, or None
        Converted angle(s) in gradians, or ``None`` when ``out`` is provided.
    """
    return np.multiply(values, 200 / 180, out=out)


def gon2rad(values: Array_Float_T | float, out: Optional[Array_Float_T] = None) -> Array_Float_T | float | None:
    """Convert gradians (gon) to radians.

    Parameters
    ----------
    values : Array_Float_T or float
        Input angle(s) in gradians.
    out : Array_Float_T, optional
        Output array for in-place writes. Default = None.

    Returns
    -------
    Array_Float_T, float, or None
        Converted angle(s) in radians, or ``None`` when ``out`` is provided.
    """
    return np.multiply(values, np.pi / 200, out=out)


def gon2deg(values: Array_Float_T | float, out: Optional[Array_Float_T] = None) -> Array_Float_T | float | None:
    """Convert gradians (gon) to degrees.

    Parameters
    ----------
    values : Array_Float_T or float
        Input angle(s) in gradians.
    out : Array_Float_T, optional
        Output array for in-place writes. Default = None.

    Returns
    -------
    Array_Float_T, float, or None
        Converted angle(s) in degrees, or ``None`` when ``out`` is provided.
    """
    return np.multiply(values, 180 / 200, out=out)


def _rad2deg(values: Array_Float_T | float, out: Optional[Array_Float_T] = None) -> Array_Float_T | float | None:
    """Delegate to :func:`rad2deg` — deprecated alias retained for backwards compatibility.

    .. deprecated:: 0.5
       The underscore alias will be removed in v0.6. Use :func:`rad2deg`.
    """
    warnings.warn(
        "GSEGUtils.util._rad2deg is deprecated; use rad2deg. The underscore alias will be removed in v0.6.",
        DeprecationWarning,
        stacklevel=2,
    )
    return rad2deg(values, out=out)


def _rad2gon(values: Array_Float_T | float, out: Optional[Array_Float_T] = None) -> Array_Float_T | float | None:
    """Delegate to :func:`rad2gon` — deprecated alias retained for backwards compatibility.

    .. deprecated:: 0.5
       The underscore alias will be removed in v0.6. Use :func:`rad2gon`.
    """
    warnings.warn(
        "GSEGUtils.util._rad2gon is deprecated; use rad2gon. The underscore alias will be removed in v0.6.",
        DeprecationWarning,
        stacklevel=2,
    )
    return rad2gon(values, out=out)


def _deg2rad(values: Array_Float_T | float, out: Optional[Array_Float_T] = None) -> Array_Float_T | float | None:
    """Delegate to :func:`deg2rad` — deprecated alias retained for backwards compatibility.

    .. deprecated:: 0.5
       The underscore alias will be removed in v0.6. Use :func:`deg2rad`.
    """
    warnings.warn(
        "GSEGUtils.util._deg2rad is deprecated; use deg2rad. The underscore alias will be removed in v0.6.",
        DeprecationWarning,
        stacklevel=2,
    )
    return deg2rad(values, out=out)


def _deg2gon(values: Array_Float_T | float, out: Optional[Array_Float_T] = None) -> Array_Float_T | float | None:
    """Delegate to :func:`deg2gon` — deprecated alias retained for backwards compatibility.

    .. deprecated:: 0.5
       The underscore alias will be removed in v0.6. Use :func:`deg2gon`.
    """
    warnings.warn(
        "GSEGUtils.util._deg2gon is deprecated; use deg2gon. The underscore alias will be removed in v0.6.",
        DeprecationWarning,
        stacklevel=2,
    )
    return deg2gon(values, out=out)


def _gon2rad(values: Array_Float_T | float, out: Optional[Array_Float_T] = None) -> Array_Float_T | float | None:
    """Delegate to :func:`gon2rad` — deprecated alias retained for backwards compatibility.

    .. deprecated:: 0.5
       The underscore alias will be removed in v0.6. Use :func:`gon2rad`.
    """
    warnings.warn(
        "GSEGUtils.util._gon2rad is deprecated; use gon2rad. The underscore alias will be removed in v0.6.",
        DeprecationWarning,
        stacklevel=2,
    )
    return gon2rad(values, out=out)


def _gon2deg(values: Array_Float_T | float, out: Optional[Array_Float_T] = None) -> Array_Float_T | float | None:
    """Delegate to :func:`gon2deg` — deprecated alias retained for backwards compatibility.

    .. deprecated:: 0.5
       The underscore alias will be removed in v0.6. Use :func:`gon2deg`.
    """
    warnings.warn(
        "GSEGUtils.util._gon2deg is deprecated; use gon2deg. The underscore alias will be removed in v0.6.",
        DeprecationWarning,
        stacklevel=2,
    )
    return gon2deg(values, out=out)


def unique_rows_fast(bin_idx: Array_Int32_T) -> tuple[ArrayT, Array_Int32_T]:
    """Determine unique rows in a 2-D integer array.

    Returns ``(unique_rows, inverse_indices)`` exactly like
    :func:`numpy.unique` with ``axis=0, return_inverse=True`` but ~5–10× faster
    for large ``N``.

    Parameters
    ----------
    bin_idx : Array_Int32_T

    Returns
    -------
    unique_rows : ArrayT,
    inverse_indices : Array_Int32_T
    """
    # make sure data is contiguous so the view trick works
    arr = np.ascontiguousarray(bin_idx)

    # view each row as a single opaque blob of bytes
    byte_dt = np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    blob = arr.view(byte_dt).ravel()

    # unique over the 1D blob array
    uniq_blob, inv = np.unique(blob, return_inverse=True)

    # turn the blobs back into an (M, D) int32 array
    uniq = uniq_blob.view(arr.dtype).reshape(-1, arr.shape[1])

    return uniq, inv  # type: ignore
