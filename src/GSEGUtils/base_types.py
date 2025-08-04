from __future__ import annotations

from typing import Annotated, Union, Sequence, Any, Optional, TypeAlias, SupportsIndex, TypedDict

import numpy as np
import numpy.typing as npt
from numpydantic import NDArray, Shape  # type: ignore[import-untyped]
from numpydantic.dtype import (  # type: ignore[import-untyped]
    Bool, Float, Integer,
    Float32, Float64,
    Int8, Int16, Int32, Int64, SignedInteger,
    UInt8, UInt16, UInt32, UnsignedInteger
)
from pydantic import StringConstraints

LowerStr = Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True)]
SfNameT = Optional[LowerStr]

ArrayDtypes = (Integer, Float, Bool)
IndexDtypes = (Integer, Bool)
BoolArrayT: TypeAlias = npt.NDArray[np.bool_]

ShapeLikeT: TypeAlias = SupportsIndex | Sequence[SupportsIndex]
NumberLikeT: TypeAlias = complex | np.number | np.bool

# ======== ARRAY TYPES ========
ArrayT = NDArray[Shape["*, ..."], ArrayDtypes]

# General dtypes
Array_Float_T = NDArray[Shape["*, ..."], Float]
Array_Integer_T = NDArray[Shape["*, ..."], Integer]
Array_SignedInteger_T = NDArray[Shape["*, ..."], SignedInteger]
Array_UnsignedInteger_T = NDArray[Shape["*, ..."], UnsignedInteger]

# Specific dtypes
Array_Float32_T = NDArray[Shape["*, ..."], Float32]
Array_Float64_T = NDArray[Shape["*, ..."], Float64]
Array_Int8_T = NDArray[Shape["*, ..."], Int8]
Array_Int16_T = NDArray[Shape["*, ..."], Int16]
Array_Int32_T = NDArray[Shape["*, ..."], Int32]
Array_Int64_T = NDArray[Shape["*, ..."], Int64]
Array_Uint8_T = NDArray[Shape["*, ..."], UInt8]
Array_Uint16_T = NDArray[Shape["*, ..."], UInt16]
Array_Uint32_T = NDArray[Shape["*, ..."], UInt32]
Array_Bool_T = NDArray[Shape["*, ..."], Bool]

# Size constrained
Array_NxM_T = NDArray[Shape["*, *"], ArrayDtypes]       # Intensity/depth image
Array_NxM_3_T = NDArray[Shape["*, *, 3"], ArrayDtypes]  # RGB image
Array_Nx2_T = NDArray[Shape["*, 2"], ArrayDtypes]       # Image coordinates
Array_Nx3_T = NDArray[Shape["*, 3"], ArrayDtypes]       # 3D Coordinates / normals
Array_3x3_T = NDArray[Shape["4, 4"], ArrayDtypes]       # Rotation Matrix
Array_4x4_T = NDArray[Shape["4, 4"], ArrayDtypes]       # Affine Transformation

# Size and type constrained
Array_Nx3_Float_T = NDArray[Shape["*, 3"], Float]
Array_Nx3_Float32_T = NDArray[Shape["*, 3"], Float32]   # Optimised coordinates
Array_Nx3_Uint8_T = NDArray[Shape["*, 3"], UInt8]       # RGB

# ======== VECTOR TYPES ========
VectorT = NDArray[Shape["*"], ArrayDtypes]              # Scalar Fields

# Datatype constrained Vector
Vector_Bool_T = NDArray[Shape["*"], Bool]               # Mask or boolean vector
Vector_Float_T = NDArray[Shape["*"], Float]
Vector_Float64_T = NDArray[Shape["*"], Float64]
Vector_Float32_T = NDArray[Shape["*"], Float32]         # Normal vector field
Vector_Int32_T = NDArray[Shape["*"], Int32]
Vector_Int16_T = NDArray[Shape["*"], Int16]
Vector_Int8_T = NDArray[Shape["*"], Int8]
Vector_Uint16_T = NDArray[Shape["*"], UInt16]           # Potential for intensity / reflectance
Vector_Uint8_T = NDArray[Shape["*"], UInt8]             # Single RGB field
Vector_IndexT = NDArray[Shape["*"], IndexDtypes]

# Shape constrained vectors
Vector_4_T = NDArray[Shape["4"], ArrayDtypes]           # FoV as vector
Vector_3_T = NDArray[Shape["3"], ArrayDtypes]           # 3D coordinate
Vector_3_Float64_T = NDArray[Shape["3"], Float64]
Vector_2_T = NDArray[Shape["2"], ArrayDtypes]           # Image coordinate

IndexLike = Union[int, slice, npt.NDArray[np.bool_], npt.NDArray[np.integer], Sequence]


class DtypeDict(TypedDict):
    names: list[LowerStr]
    formats: list[npt.DTypeLike]


def make_ndarray_type(
        *dimensions: Optional[int | str],
        dtype: Optional[npt.DTypeLike] = None
) -> type[NDArray[Any, Any]]:
    """
    Helper function to _generate the numpydantic type for a ndarray.

    Calling 'make_ndarray_type(None, 3, dtype=np.float32)' would return a numpydantic dtype corresponding to an array
    of shape (N, 3) with dtype = np.float32 and would provide pydantic validation on this
    """
    if len(dimensions) == 0:
        shape_list = ["*", "..."]
    else:
        shape_list = [str(x) if x is not None else "*" for x in dimensions]

    result : type[NDArray[Any, Any]] = NDArray[Shape[", ".join(shape_list)], dtype if dtype is not None else Any]

    return result