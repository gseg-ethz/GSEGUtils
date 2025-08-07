from __future__ import annotations

from typing import Annotated, Union, Sequence, Any, Optional, TypeAlias, SupportsIndex, TypedDict

import numpy as np
import numpy.typing as npt
from numpydantic import NDArray, Shape  # type: ignore[import-untyped]
from numpydantic.types import NDArrayType
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
ArrayT =                        NDArray[Shape["*, ..."], ArrayDtypes]
"""
| *[int, ...] Generic NDArray type* 
|
| Additional specific dtyped definitions: 

    * **Array_Float_T**
    * **Array_Integer_T**
    * **Array_SignedInteger_T**
    * **Array_UnsignedInteger_T**
    * **Array_Bool_T**
    * **Array_Float32_T**
    * **Array_Float64_T**
    * **Array_Int8_T**
    * **Array_Int16_T**
    * **Array_Int32_T**
    * **Array_Int64_T**
    * **Array_Uint8_T**
    * **Array_Uint16_T**
    * **Array_Uint32_T**

"""

# Generalised dtypes
Array_Float_T =                 NDArray[Shape["*, ..."], Float]             #: See :any:`ArrayT`
Array_Integer_T =               NDArray[Shape["*, ..."], Integer]           #: See :any:`ArrayT`
Array_SignedInteger_T =         NDArray[Shape["*, ..."], SignedInteger]     #: See :any:`ArrayT`
Array_UnsignedInteger_T =       NDArray[Shape["*, ..."], UnsignedInteger]   #: See :any:`ArrayT`
Array_Bool_T =                  NDArray[Shape["*, ..."], Bool]              #: See :any:`ArrayT`
Array_Float32_T =               NDArray[Shape["*, ..."], Float32]           #: See :any:`ArrayT`
Array_Float64_T =               NDArray[Shape["*, ..."], Float64]           #: See :any:`ArrayT`
Array_Int8_T =                  NDArray[Shape["*, ..."], Int8]              #: See :any:`ArrayT`
Array_Int16_T =                 NDArray[Shape["*, ..."], Int16]             #: See :any:`ArrayT`
Array_Int32_T =                 NDArray[Shape["*, ..."], Int32]             #: See :any:`ArrayT`
Array_Int64_T =                 NDArray[Shape["*, ..."], Int64]             #: See :any:`ArrayT`
Array_Uint8_T =                 NDArray[Shape["*, ..."], UInt8]             #: See :any:`ArrayT`
Array_Uint16_T =                NDArray[Shape["*, ..."], UInt16]            #: See :any:`ArrayT`
Array_Uint32_T =                NDArray[Shape["*, ..."], UInt32]            #: See :any:`ArrayT`

# Size constrained
Array_NxM_T =                   NDArray[Shape["*, *"], ArrayDtypes]       # Intensity/depth image
"""
| [NxM] Generic NDArray type
|
| Additional specific dtyped definitions:

    * **Array_NxM_Float_T**
    * **Array_NxM_Integer_T**
    * **Array_NxM_SignedInteger_T**
    * **Array_NxM_UnsignedInteger_T**
    * **Array_NxM_Bool_T**
    * **Array_NxM_Float32_T**
    * **Array_NxM_Float64_T**
    * **Array_NxM_Int8_T**
    * **Array_NxM_Int16_T**
    * **Array_NxM_Int32_T**
    * **Array_NxM_Int64_T**
    * **Array_NxM_Uint8_T**
    * **Array_NxM_Uint16_T**
    * **Array_NxM_Uint32_T**

"""

Array_NxM_Float_T =             NDArray[Shape["*, *"], Float]           #: See :any:`Array_NxM_T`
Array_NxM_Integer_T =           NDArray[Shape["*, *"], Integer]         #: See :any:`Array_NxM_T`
Array_NxM_SignedInteger_T =     NDArray[Shape["*, *"], SignedInteger]   #: See :any:`Array_NxM_T`
Array_NxM_UnsignedInteger_T =   NDArray[Shape["*, *"], UnsignedInteger] #: See :any:`Array_NxM_T`
Array_NxM_Bool_T =              NDArray[Shape["*, *"], Bool]            #: See :any:`Array_NxM_T`
Array_NxM_Float32_T =           NDArray[Shape["*, *"], Float32]         #: See :any:`Array_NxM_T`
Array_NxM_Float64_T =           NDArray[Shape["*, *"], Float64]         #: See :any:`Array_NxM_T`
Array_NxM_Int8_T =              NDArray[Shape["*, *"], Int8]            #: See :any:`Array_NxM_T`
Array_NxM_Int16_T =             NDArray[Shape["*, *"], Int16]           #: See :any:`Array_NxM_T`
Array_NxM_Int32_T =             NDArray[Shape["*, *"], Int32]           #: See :any:`Array_NxM_T`
Array_NxM_Int64_T =             NDArray[Shape["*, *"], Int64]           #: See :any:`Array_NxM_T`
Array_NxM_Uint8_T =             NDArray[Shape["*, *"], UInt8]           #: See :any:`Array_NxM_T`
Array_NxM_Uint16_T =            NDArray[Shape["*, *"], UInt16]          #: See :any:`Array_NxM_T`
Array_NxM_Uint32_T =            NDArray[Shape["*, *"], UInt32]          #: See :any:`Array_NxM_T`

# Typical 3 Channel Image
Array_NxM_3_T =                 NDArray[Shape["*, *, 3"], ArrayDtypes]  # RGB image
"""
| [NxMx3] Generic NDArray type 
| Ideal for supporting RGB images
|
| Additional specific dtyped definitions:

    * **Array_NxMx3_Uint8_T**

"""
Array_NxM_3_Uint8_T =           NDArray[Shape["*, *, 3"], UInt8]  # RGB image

# Nx2 Constrained - e.g. coordinate pairs
Array_Nx2_T =                   NDArray[Shape["*, 2"], ArrayDtypes]
"""
| [Nx2] Generic NDArray type
| Ideal for supporting coordinate pairs
|
| Additional specific dtyped definitions:

    * **Array_Nx2_Float_T**
    * **Array_Nx2_Integer_T**
    * **Array_Nx2_SignedInteger_T**
    * **Array_Nx2_UnsignedInteger_T**
    * **Array_Nx2_Bool_T**
    * **Array_Nx2_Float32_T**
    * **Array_Nx2_Float64_T**
    * **Array_Nx2_Int8_T**
    * **Array_Nx2_Int16_T**
    * **Array_Nx2_Int32_T**
    * **Array_Nx2_Int64_T**
    * **Array_Nx2_Uint8_T**
    * **Array_Nx2_Uint16_T**
    * **Array_Nx2_Uint32_T**

"""
# Nx2 Generalised dtypes
Array_Nx2_Float_T =             NDArray[Shape["*, 2"], Float]           #: See :any:`Array_Nx2_T`
Array_Nx2_Integer_T =           NDArray[Shape["*, 2"], Integer]         #: See :any:`Array_Nx2_T`
Array_Nx2_SignedInteger_T =     NDArray[Shape["*, 2"], SignedInteger]   #: See :any:`Array_Nx2_T`
Array_Nx2_UnsignedInteger_T =   NDArray[Shape["*, 2"], UnsignedInteger] #: See :any:`Array_Nx2_T`
Array_Nx2_Bool_T =              NDArray[Shape["*, 2"], Bool]            #: See :any:`Array_Nx2_T`
Array_Nx2_Float32_T =           NDArray[Shape["*, 2"], Float32]         #: See :any:`Array_Nx2_T`
Array_Nx2_Float64_T =           NDArray[Shape["*, 2"], Float64]         #: See :any:`Array_Nx2_T`
Array_Nx2_Int8_T =              NDArray[Shape["*, 2"], Int8]            #: See :any:`Array_Nx2_T`
Array_Nx2_Int16_T =             NDArray[Shape["*, 2"], Int16]           #: See :any:`Array_Nx2_T`
Array_Nx2_Int32_T =             NDArray[Shape["*, 2"], Int32]           #: See :any:`Array_Nx2_T`
Array_Nx2_Int64_T =             NDArray[Shape["*, 2"], Int64]           #: See :any:`Array_Nx2_T`
Array_Nx2_Uint8_T =             NDArray[Shape["*, 2"], UInt8]           #: See :any:`Array_Nx2_T`
Array_Nx2_Uint16_T =            NDArray[Shape["*, 2"], UInt16]          #: See :any:`Array_Nx2_T`
Array_Nx2_Uint32_T =            NDArray[Shape["*, 2"], UInt32]          #: See :any:`Array_Nx2_T`

# Nx3 - e.g. Cartesian Coordinates, Normal Vectors, RGB fields
Array_Nx3_T =                   NDArray[Shape["*, 3"], ArrayDtypes]
"""
| [Nx3] Generic NDArray type
| Ideal for supporting Cartesian coordinates, normal vectors, RGB fields
|
| Additional specific dtyped definitions:

    * **Array_Nx3_Float_T**
    * **Array_Nx3_Integer_T**
    * **Array_Nx3_SignedInteger_T**
    * **Array_Nx3_UnsignedInteger_T**
    * **Array_Nx3_Bool_T**
    * **Array_Nx3_Float32_T**
    * **Array_Nx3_Float64_T**
    * **Array_Nx3_Int8_T**
    * **Array_Nx3_Int16_T**
    * **Array_Nx3_Int32_T**
    * **Array_Nx3_Int64_T**
    * **Array_Nx3_Uint8_T**
    * **Array_Nx3_Uint16_T**
    * **Array_Nx3_Uint32_T**
    
"""
# Nx3 Generalised dtypes
Array_Nx3_Float_T =             NDArray[Shape["*, 3"], Float]           #: See :any:`Array_Nx3_T`
Array_Nx3_Integer_T =           NDArray[Shape["*, 3"], Integer]         #: See :any:`Array_Nx3_T`
Array_Nx3_SignedInteger_T =     NDArray[Shape["*, 3"], SignedInteger]   #: See :any:`Array_Nx3_T`
Array_Nx3_UnsignedInteger_T =   NDArray[Shape["*, 3"], UnsignedInteger] #: See :any:`Array_Nx3_T`
Array_Nx3_Bool_T =              NDArray[Shape["*, 3"], Bool]            #: See :any:`Array_Nx3_T`

# Nx3 Specific dtypes
Array_Nx3_Float32_T =           NDArray[Shape["*, 3"], Float32]         #: See :any:`Array_Nx3_T`
Array_Nx3_Float64_T =           NDArray[Shape["*, 3"], Float64]         #: See :any:`Array_Nx3_T`
Array_Nx3_Int8_T =              NDArray[Shape["*, 3"], Int8]            #: See :any:`Array_Nx3_T`
Array_Nx3_Int16_T =             NDArray[Shape["*, 3"], Int16]           #: See :any:`Array_Nx3_T`
Array_Nx3_Int32_T =             NDArray[Shape["*, 3"], Int32]           #: See :any:`Array_Nx3_T`
Array_Nx3_Int64_T =             NDArray[Shape["*, 3"], Int64]           #: See :any:`Array_Nx3_T`
Array_Nx3_Uint8_T =             NDArray[Shape["*, 3"], UInt8]           #: See :any:`Array_Nx3_T`
Array_Nx3_Uint16_T =            NDArray[Shape["*, 3"], UInt16]          #: See :any:`Array_Nx3_T`
Array_Nx3_Uint32_T =            NDArray[Shape["*, 3"], UInt32]          #: See :any:`Array_Nx3_T`

# ======== TRANSFORMATION MATRICES / Rotation Matrices ========
# Rotation Matrix / Camera Matrix
Array_3x3_T =                   NDArray[Shape["3, 3"], ArrayDtypes]
"""
| [3x3] Generic NDArray type
| Ideal for supporting rotation matrices and camera projection matrices
|
| Additional specific dtyped definitions:

    * **Array_3x3_Float_T**
    * **Array_3x3_Float32_T**
    * **Array_3x3_Float64_T**

"""
Array_3x3_Float_T =             NDArray[Shape["3, 3"], Float]           #: See :any:`Array_3x3_T`
Array_3x3_Float32_T =           NDArray[Shape["3, 3"], Float32]         #: See :any:`Array_3x3_T`
Array_3x3_Float64_T =           NDArray[Shape["3, 3"], Float64]         #: See :any:`Array_3x3_T`

# Affine Transform Matrix
Array_4x4_T =                   NDArray[Shape["4, 4"], ArrayDtypes]
"""
| [4x4] Generic NDArray type
| Ideal for affine transformation matrices to apply to homogeneous 3D coordinates
|
| Additional specific dtyped definitions:

    * **Array_4x4_Float_T**
    * **Array_4x4_Float32_T**
    * **Array_4x4_Float64_T**

"""
Array_4x4_Float_T =             NDArray[Shape["4, 4"], Float]           #: See :any:`Array_4x4_T`
Array_4x4_Float32_T =           NDArray[Shape["4, 4"], Float32]         #: See :any:`Array_4x4_T`
Array_4x4_Float64_T =           NDArray[Shape["4, 4"], Float64]         #: See :any:`Array_4x4_T`

# ======== VECTOR TYPES ========
VectorT =                       NDArray[Shape["*"], ArrayDtypes]
"""
| [N,] Generic Vector type
| Ideal for supporting Scalar Fields, Indexes, Boolean masks and Segmentation classification
|
| Additional specific dtyped definitions:

    * **Vector_Float_T**
    * **Vector_Integer_T**
    * **Vector_SignedInteger_T**
    * **Vector_UnsignedInteger_T**
    * **Vector_Bool_T**
    * **Vector_Float32_T**
    * **Vector_Float64_T**
    * **Vector_Int8_T**
    * **Vector_Int16_T**
    * **Vector_Int32_T**
    * **Vector_Int64_T**
    * **Vector_Uint8_T**
    * **Vector_Uint16_T**
    * **Vector_Uint32_T**

"""
# Generalised Dtypes
Vector_Float_T =                NDArray[Shape["*"], Float]              #: See :any:`VectorT`
Vector_Integer_T =              NDArray[Shape["*"], Integer]            #: See :any:`VectorT`
Vector_SignedInteger_T =        NDArray[Shape["*"], SignedInteger]      #: See :any:`VectorT`
Vector_UnsignedInteger_T =      NDArray[Shape["*"], UnsignedInteger]    #: See :any:`VectorT`
Vector_Bool_T =                 NDArray[Shape["*"], Bool]               #: See :any:`VectorT`
Vector_Float32_T =              NDArray[Shape["*"], Float32]            #: See :any:`VectorT`
Vector_Float64_T =              NDArray[Shape["*"], Float64]            #: See :any:`VectorT`
Vector_Int8_T =                 NDArray[Shape["*"], Int8]               #: See :any:`VectorT`
Vector_Int16_T =                NDArray[Shape["*"], Int16]              #: See :any:`VectorT`
Vector_Int32_T =                NDArray[Shape["*"], Int32]              #: See :any:`VectorT`
Vector_Int64_T =                NDArray[Shape["*"], Int64]              #: See :any:`VectorT`
Vector_Uint8_T =                NDArray[Shape["*"], UInt8]              #: See :any:`VectorT`
Vector_Uint16_T =               NDArray[Shape["*"], UInt16]             #: See :any:`VectorT`
Vector_Uint32_T =               NDArray[Shape["*"], UInt32]             #: See :any:`VectorT`

Vector_IndexT =                 NDArray[Shape["*"], IndexDtypes]
""" Special Vector type which supports integer or bool dtypes."""

# ======== 3D POINT, NORMAL VECTOR, RGB VALUE ========
Vector_3_T =                    NDArray[Shape["3"], ArrayDtypes]
"""
| [3,] Generic 3 Element Vector type
| Useful for single 3D coordinates, RGB values or other 3D Vectors
|
| Additional specific dtyped definitions:

    * **Vector_Float_T**
    * **Vector_Integer_T**
    * **Vector_SignedInteger_T**
    * **Vector_UnsignedInteger_T**
    * **Vector_Bool_T**
    * **Vector_Float32_T**
    * **Vector_Float64_T**
    * **Vector_Int8_T**
    * **Vector_Int16_T**
    * **Vector_Int32_T**
    * **Vector_Int64_T**
    * **Vector_Uint8_T**
    * **Vector_Uint16_T**
    * **Vector_Uint32_T**

"""
# Generalised Dtypes
Vector_3_Float_T =              NDArray[Shape["3"], Float]              #: See :any:`Vector_3_T`
Vector_3_Integer_T =            NDArray[Shape["3"], Integer]            #: See :any:`Vector_3_T`
Vector_3_SignedInteger_T =      NDArray[Shape["3"], SignedInteger]      #: See :any:`Vector_3_T`
Vector_3_UnsignedInteger_T =    NDArray[Shape["3"], UnsignedInteger]    #: See :any:`Vector_3_T`
Vector_3_Bool_T =               NDArray[Shape["3"], Bool]               #: See :any:`Vector_3_T`
Vector_3_Float32_T =            NDArray[Shape["3"], Float32]            #: See :any:`Vector_3_T`
Vector_3_Float64_T =            NDArray[Shape["3"], Float64]            #: See :any:`Vector_3_T`
Vector_3_Int8_T =               NDArray[Shape["3"], Int8]               #: See :any:`Vector_3_T`
Vector_3_Int16_T =              NDArray[Shape["3"], Int16]              #: See :any:`Vector_3_T`
Vector_3_Int32_T =              NDArray[Shape["3"], Int32]              #: See :any:`Vector_3_T`
Vector_3_Int64_T =              NDArray[Shape["3"], Int64]              #: See :any:`Vector_3_T`
Vector_3_Uint8_T =              NDArray[Shape["3"], UInt8]              #: See :any:`Vector_3_T`
Vector_3_Uint16_T =             NDArray[Shape["3"], UInt16]             #: See :any:`Vector_3_T`
Vector_3_Uint32_T =             NDArray[Shape["3"], UInt32]             #: See :any:`Vector_3_T`

# ======== POINT LIKE ========
Vector_4_T =                    NDArray[Shape["4"], ArrayDtypes]    #: Vector of size 4
Vector_2_T =                    NDArray[Shape["2"], ArrayDtypes]    #: Vector of size 2

IndexLike = Union[int, slice, npt.NDArray[np.bool_], npt.NDArray[np.integer], Sequence]


class DtypeDict(TypedDict):
    names: list[LowerStr]
    formats: list[npt.DTypeLike]


def make_ndarray_type(
        *dimensions: Optional[int | str],
        dtype: Optional[npt.DTypeLike] = None
) -> NDArrayType:
    """
    Helper function to _generate the numpydantic type for a ndarray.

    Calling 'make_ndarray_type(None, 3, dtype=np.float32)' would return a numpydantic dtype corresponding to an array
    of shape (N, 3) with dtype = np.float32 and would provide pydantic validation on this
    """
    if len(dimensions) == 0:
        shape_list = ["*", "..."]
    else:
        shape_list = [str(x) if x is not None else "*" for x in dimensions]

    return NDArray[Shape[", ".join(shape_list)], dtype if dtype is not None else Any]