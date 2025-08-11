""" GSEGUtils.util
This module provides utility functions and constants for angle conversion and numerical operations,
along with an enumeration for specifying angle units.
"""

import logging
from enum import StrEnum
from typing import Any, Optional, cast

import numpy as np
import numpy.typing as npt

from .base_types import ArrayT, Array_Int32_T, Array_Float_T

logger = logging.getLogger(__name__.split(".")[0])


class AngleUnit(StrEnum):
    """
    An enumeration for angular units.

    * AngleUnit.RAD = 'rad'
    * AngleUnit.DEGREE = 'deg'
    * AngleUnit.GON = 'gon'

    """

    RAD = "rad"
    DEGREE = "deg"
    GON = "gon"


def convert_angles(
        values: Array_Float_T,
        source_unit: AngleUnit,
        target_unit: AngleUnit,
        out: Optional[Array_Float_T] = None
) -> Array_Float_T|None:
    """
    Converts an array of angles from one unit to another.

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
            logger.warning(f"Input values are not an ndarray, returning None and not assigning converted values")
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
            return _rad2deg(values) if out is None else _rad2deg(values, out=out)
        else:
            return _rad2gon(values, out=out)

    elif source_unit == AngleUnit.DEGREE:
        if target_unit == AngleUnit.RAD:
            return _deg2rad(values, out=out)
        else:
            return _deg2gon(values, out=out)

    else:
        if target_unit == AngleUnit.RAD:
            return _gon2rad(values, out=out)
        else:
            return _gon2deg(values, out=out)

# TODO updated naming to not be private
def _rad2deg(values: Array_Float_T|float, out: Optional[Array_Float_T]=None) -> Array_Float_T|float|None:
    """ Convert radians to degrees.

    Parameters
    ----------
    values : Array_Float_T|float
    out : Optional[Array_Float_T], default=None

    Returns
    -------
    Array_Float_T|float|None
    """
    return np.rad2deg(values, out=out)

def _rad2gon(values: Array_Float_T|float, out: Optional[Array_Float_T]=None) -> Array_Float_T|float|None:
    """ Convert radians to gradians(gon).

    Parameters
    ----------
    values : Array_Float_T|float
    out : Optional[Array_Float_T], default=None

    Returns
    -------
    Array_Float_T|float|None
    """
    return np.multiply(values, 200 / np.pi, out=out)

def _deg2rad(values: Array_Float_T|float, out: Optional[Array_Float_T]=None) -> Array_Float_T|float|None:
    """ Convert degrees to radians.

    Parameters
    ----------
    values : Array_Float_T|float
    out : Optional[Array_Float_T], default=None

    Returns
    -------
    Array_Float_T|float|None
    """
    return np.deg2rad(values, out=out)

def _deg2gon(values: Array_Float_T|float, out: Optional[Array_Float_T]=None) -> Array_Float_T|float|None:
    """ Convert degrees to gradians(gon).

    Parameters
    ----------
    values : Array_Float_T|float
    out : Optional[Array_Float_T], default=None

    Returns
    -------
    Array_Float_T|float|None
    """
    return np.multiply(values, 200 / 180, out=out)

def _gon2rad(values: Array_Float_T|float, out: Optional[Array_Float_T]=None) -> Array_Float_T|float|None:
    """ Convert gradians(gon) to radians.

    Parameters
    ----------
    values : Array_Float_T|float
    out : Optional[Array_Float_T], default=None

    Returns
    -------
    Array_Float_T|float|None
    """
    return np.multiply(values, np.pi / 200, out=out)

def _gon2deg(values: Array_Float_T|float, out: Optional[Array_Float_T]=None) -> Array_Float_T|float|None:
    """ Convert gradians(gon) to degrees.

    Parameters
    ----------
    values : Array_Float_T|float
    out : Optional[Array_Float_T], default=None

    Returns
    -------
    Array_Float_T|float|None
    """
    return np.multiply(values, 180 / 200, out=out)

def unique_rows_fast(bin_idx: Array_Int32_T) -> tuple[ArrayT, Array_Int32_T]:
    """ Determine unique rows in a 2D array of integers.
    Returns `(unique_rows, inverse_indices)` exactly like `np.unique(bin_idx, axis=0, return_inverse=True)`
    but ~5–10× faster for large N.

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

    return uniq, inv    # type: ignore
