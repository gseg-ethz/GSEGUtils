"""
    Help on Base Arrays.

    NAME
        base_arrays

    DESCRIPTION
        Subclassable base array module for better handling of array data
        ================================================================

        The base_arrays module is designed to create sub-classable array like objects that:

        * **Looks like a Numpy array. Sounds like a Numpy array. ACTS like a Numpy array!**
        * **Automated data validation (including array shape and dtype) with Pydantic**
        * **Support for extra attribute definition and access like a dataclass**
        * **Easy to subclass whilst getting the same functionality**

        This module contains the following base array types which can be subclassed as needed:

        * *BaseArray*: Base class for all other array types
        * *NumericMixins*: Extends numpy like magic method behaviour for equality and numeric operations
        * *FixedLengthArray*: Treats the array as a list of row vectors
        * *BaseVector*: 1D Vector
        * *HomogeneousArray*: Arrays that can be converted to homogeneous coordinates
        * *ArrayNx2*: Nx2 sized arrays (E.g. 2D coordinates)
        * *ArrayNx3*: Nx3 sized arrays (E.g. 3D coordinates)
"""

from __future__ import annotations

import logging
import builtins
import copy
from abc import ABC
from typing import (
    Any,
    Generator,
    MutableMapping,
    Optional,
    Self,
    TypedDict,
    NotRequired,
    Unpack,
    TypeVar,
    cast
)

import numpy as np
import numpy.typing as npt
from numpydantic import NDArray # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, field_validator, ValidationError

from .validators import validate_transposed_2d_array, convert_slice_to_integer_range
from .base_types import (
    ArrayT,
    Array_Nx2_T,
    Array_Nx3_T,
    Array_Nx3_Float_T,
    IndexLike,
    Vector_IndexT,
    VectorT,
    BoolArrayT,
    NumberLikeT
)


# noinspection PyProtectedMember
from numpy._typing._array_like import _ArrayLikeBool_co


class MinMaxKwargsT(TypedDict, total=False):
    out: NotRequired[None]
    keepdims: NotRequired[builtins.bool]
    initial: NotRequired[NumberLikeT]
    where: NotRequired[_ArrayLikeBool_co]


SelfT = TypeVar('SelfT', bound='BaseArray')

logger = logging.getLogger(__name__)


__all__ = ['BaseArray', 'NumericMixins', 'FixedLengthArray', 'BaseVector', 'HomogeneousArray', 'ArrayNx2', 'ArrayNx3']

class BaseArray(ABC, BaseModel):
    """ A sub-classable, automatically validated array class.

    BaseArray performs no shape validation but will check that the array is of
    np.bool_, np.floating or np.integer types.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,   # Required for the numpy types
        validate_assignment=True,       # Should validate anytime an attribute is set
        revalidate_instances="never",   # Don't keep validating instances (avoids infinite validation loops)
        validate_default=True,          # Ensure default values get validated as well
        strict=True,                    # Ensure that no coercion of types occurs - strict typechecking
        frozen=False,                   # Object can be manipulated
        extra="ignore",                 # Extra fields passed are not stored in the object (e.g. kwargs)
        serialize_by_alias=False,       # Serialisation takes the original field names (e.g. 'arr')
        populate_by_name=False,         # Field is not expected to be populated by attribute name if an alias exists
    )

    arr: ArrayT #: Object attribute that contains all raw numpy data object

    def __init__(self, arr: ArrayT, **kwargs: dict[str, Any]):
        super().__init__(arr=arr, **kwargs)

    # noinspection PyNestedDecorators
    @field_validator('arr', mode='before')
    @classmethod
    def _coerce_array(cls, value: npt.ArrayLike) -> ArrayT:
        """

        Parameters
        ----------
        value: npt.ArrayLike

        Returns
        -------
        ArrayT
        """
        if isinstance(value, BaseArray):
            value = value.arr

        return np.atleast_1d(np.asarray(value))

    @property
    def __array_interface__(self) -> dict[str, Any]:
        """
            Access to the base __array_interface__ property.

            `__array_interface__` allows interaction with the object from other
            libraries and functions that support the array interface protocol.

            Returns
            -------
            dict[str, Any]
                Represents the array's metadata and structure
        """
        return self.arr.__array_interface__

    # noinspection PyPep8Naming
    @property
    def T(self) -> ArrayT:
        """ Transposed view of the array.

        Returns
        -------
        ArrayT
        """
        return self.arr.T

    @property
    def shape(self) -> tuple[int, ...]:
        """ Shape of the array

        Returns
        -------
        tuple[int, ...]
        """
        return self.arr.shape

    @property
    def dtype(self) -> npt.DTypeLike:
        """ Dtype of the array

        Returns
        -------
        npt.DtypeLike
        """
        return cast(np.dtype, self.arr.dtype)

    @property
    def ndim(self) -> int:
        """ Number of dimensions in the array """
        return cast(int, self.arr.ndim)

    @property
    def base(self) -> ArrayT|None:
        """ Returns the base array if the current `arr` attribute is a view. """
        return self.arr.base

    @property
    def size(self) -> int:
        """ Size of the array (number of elements) """
        return self.arr.size

    def view(self, dtype: npt.DTypeLike = None, _type: type|None = None) -> ArrayT:
        """ Get a ndarray view of the underlying array. """
        dtype = self.dtype if dtype is None else dtype

        _type = type(self.arr) if _type is None else _type

        return self.arr.view(dtype=dtype, type=_type)

    def min(self, **kwargs: Unpack[MinMaxKwargsT]) -> Any:
        """
        Computes the minimum value along the specified axis of an array, considering only
        the elements where the condition in `where` is True. The returned value or array
        can optionally retain reduced dimensions depending on the `keepdims` parameter.

        Parameters
        ----------
        **kwargs : dict of str to Any
            Optional arguments passed to the array's max function, such as `axis` for specifying
            the axis along which the maximum is computed and other relevant parameters.


        Returns
        -------
        Any
            The minimum value of the array along the specified axis or axes, optionally
            reduced as per the `keepdims` parameter, and restricted by the `where` condition.
        """
        return self.arr.min(**kwargs)

    def max(self, **kwargs: Unpack[MinMaxKwargsT]) -> Any:
        """
        Find the maximum value in the array.

        This method computes the maximum value in the array or along a specified axis, depending
        on the arguments passed. Additional arguments can be supplied to control the behaviour,
        such as axis, keepdims, or initial value.

        Parameters
        ----------
        **kwargs : dict of str to Any
            Optional arguments passed to the array's max function, such as `axis` for specifying
            the axis along which the maximum is computed and other relevant parameters.

        Returns
        -------
        Any
            The maximum value from the array, or a reduced result if an axis is provided.
        """
        return self.arr.max(**kwargs)

    def __len__(self) -> int:
        """
        Returns the number of rows in the object.

        This method returns the first dimension size of the object,
        indicating the number of rows or elements along that dimension.

        Returns
        -------
        int
            The number of rows in the object.
        """
        return self.shape[0]

    def __getitem__(self, key: IndexLike) -> ArrayT | Self:
        """
        Retrieves an element or a slice from the array based on the provided key.

        This method attempts to return a copy of the array element or slice when possible.
        If the operation fails due to a validation error, it directly retrieves the
        element or slice from the array.

        Parameters
        ----------
        key : IndexLike
            The index or slice used to retrieve an item or subset from the array.

        Returns
        -------
        ArrayT or Self
            Returns a copy of the array element or slice if no exceptions occur.
            If a validation error occurs, it directly returns the corresponding
            item or slice from the array.
        """
        try:
            return self.copy(array=self.arr[key], deep=False)
        except ValidationError:
            return self.arr[key]

    def __setitem__(self, key: IndexLike, value: ArrayT | BaseArray) -> None:
        """
        Sets the value at the given index in the array.

        This method allows item assignment in the array. If the provided
        value is an instance of BaseArray, its internal array is used
        as the value to assign. Otherwise, the value itself is assigned.

        Parameters
        ----------
        key : IndexLike
            The index at which the value is to be set. It supports any valid
            indexing type.
        value : ArrayT or BaseArray
            The value to be assigned. If the value is of type BaseArray,
            its internal array is extracted and assigned.
        """
        self.arr[key] = value.arr if isinstance(value, BaseArray) else value

    def __lt__(self, other: Any) -> BoolArrayT:
        """
        Compares elements of the array with another value or array, returning a boolean
        array indicating whether each element of the array is less than the
        corresponding element or value.

        Parameters
        ----------
        other : Any
            The value or array to compare against. Can be a scalar or an array-like
            object of the same shape as `self.arr`.

        Returns
        -------
        BoolArrayT
            A boolean array (`BoolArrayT`) of the same shape as `self.arr`, where each
            element represents the result of comparing the corresponding element of
            `self.arr` with `other`.
        """
        return self.arr < other

    def __le__(self, other: Any) -> BoolArrayT:
        """
        Compare the elements of the array with `other` using the <= operator.

        An element-wise comparison between the stored array
        and the provided `other` parameter. The result of this comparison
        is a boolean array indicating where the condition is satisfied.

        Parameters
        ----------
        other : Any
            The value or object to compare with

        Returns
        -------
        BoolArrayT
            A boolean array
        """
        return self.arr <= other

    def __ge__(self, other: Any) -> BoolArrayT:
        """
        Compares the current array with another value element-wise, determining if each element in the
        current array is greater than or equal to the corresponding element in the other value. This
        operation supports broadcasting if the shapes of the arrays are compatible.

        Parameters
        ----------
        other : Any
            The value or array to compare against. Must be broadcast-compatible with the current array.

        Returns
        -------
        BoolArrayT
            A boolean array where each element indicates True if the condition (current array >= other)
            is met, and False otherwise.
        """
        return self.arr >= other

    def __gt__(self, other: Any) -> BoolArrayT:
        """
        Compares the elements of the instance array with the provided value using the
        greater-than operator. Returns a boolean array indicating whether each element
        is greater than the provided value.

        Parameters
        ----------
        other : Any
            The value to compare each element of the instance array with.

        Returns
        -------
        BoolArrayT
            A boolean array where each element indicates the result of the greater-than
            comparison for the corresponding element of the instance array and the
            provided value.
        """
        return self.arr > other

    def __eq__(self, other: Any) -> Any: # type: ignore[override]
        """
        Compares the current object's array elements with the given object and returns
        a boolean array indicating element-wise equality.

        Parameters
        ----------
        other : Any
            The object to compare each element of the array with.

        Returns
        -------
        BoolArrayT
            A boolean array where each element is `True` if the corresponding element
            in the current object's array matches the given object, `False` otherwise.
        """
        return self.arr == other

    def __ne__(self, other: Any) -> Any: # type: ignore[override]
        """
        Compares the current object with another object for inequality.

        This method performs an element-wise inequality comparison between the array
        in the current object and the `other` object provided as an argument. It casts
        the result to the expected type and returns it.

        Parameters
        ----------
        other : Any
            The object to compare with the current object's array for inequality.

        Returns
        -------
        BoolArrayT
            A boolean array where each element indicates whether the corresponding
            elements of the object's array and `other` are not equal.
        """
        return self.arr != other

    def copy(self: Self,    # type: ignore[override]
             array: npt.NDArray[Any] | Self | None = None,
             *,
             deep: bool = True,
             update: Optional[MutableMapping[str, Any]] = None,
             **kwargs: dict[str, Any]) -> Self:
        """
        Creates a copy of the current instance with optional modifications to its data.

        This method creates and returns a new instance of the current class,
        optionally overriding the attributes specified in `update` or replacing the
        array data using the `array` parameter. If `deep` is True, a deep copy of
        the current instance's data is created, ensuring that changes to the copy
        do not affect the original instance.

        Parameters
        ----------
        array : npt.NDArray[Any] | BaseArray | None, optional
            An array to override the current instance's array data. If None, no
            changes are applied to the `arr` attribute unless specified in `update`.
        deep : bool, optional
            Whether to create a deep copy of the current instance's data. Defaults
            to True.
        update : MutableMapping[str, Any] or None, optional
            A dictionary of key-value pairs to override the instance's attributes.
            Uses "arr" as a key to specifically update the array data if provided.
        kwargs : dict[str, Any]
            Additional attributes or configurations to be passed during the creation
            of the copied instance.

        Returns
        -------
        Self
            A new instance of the same class as the current object, containing the
            updated and/or copied data.
        """
        update = update or {}

        if array is not None:
            update["arr"] = array

        if "arr" in update:
            if isinstance(update["arr"], BaseArray):
                update["arr"] = update["arr"].arr

        data = self.model_dump(exclude=set(update.keys()), by_alias=False)
        data = copy.deepcopy(data) if deep else data

        data.update(update)

        return type(self)(**data)


class NumericMixins(BaseArray):
    def __init__(self, arr: ArrayT, **kwargs: dict[str, Any]):
        super().__init__(arr=arr, **kwargs)

    def __add__(self, other: Any) -> Self:
        return self.copy(self.arr + other)

    def __sub__(self, other: Any) -> Self:
        return self.copy(self.arr - other)

    def __mul__(self, other: Any) -> Self:
        return self.copy(self.arr * other)

    def __truediv__(self, other: Any) -> Self:
        return self.copy(self.arr / other)

    def __floordiv__(self, other: Any) -> Self:
        return self.copy(self.arr // other)

    def __mod__(self, other: Any) -> Self:
        return self.copy(self.arr % other)

    def __pow__(self, other: Any) -> Self:
        return self.copy(self.arr**other)

    def __matmul__(self, other: Any) -> Self:
        return self.copy(self.arr @ other)

    def __rmatmul__(self, other: Any) -> Self:
        return self.copy(other @ self.arr)

    def __radd__(self, other: Any) -> Self:
        return self.copy(other + self.arr)

    def __rsub__(self, other: Any) -> Self:
        return self.copy(other - self.arr)

    def __rmul__(self, other: Any) -> Self:
        return self.copy(other * self.arr)

    def __rpow__(self, other: Any) -> Self:
        return self.copy(other**self.arr)

    def __rtruediv__(self, other: Any) -> Self:
        return self.copy(other / self.arr)

    def __rfloordiv__(self, other: Any) -> Self:
        return self.copy(other // self.arr)

    def __rmod__(self, other: Any) -> Self:
        return self.copy(other % self.arr)

    def __divmod__(self, other: npt.ArrayLike) -> tuple[Self, Self]:
        quotient, remainder = np.divmod(self.arr, other)
        return self.copy(quotient), self.copy(remainder)

    def __rdivmod__(self, other: Any) -> tuple[Self, Self]:
        quotient, remainder = np.divmod(other, self.arr)
        return self.copy(quotient), self.copy(remainder)

    def __neg__(self) -> Self:
        return self.copy(-self.arr)

    def __abs__(self) -> Self:
        return self.copy(np.abs(self.arr))

    def __iadd__(self, other: Any) -> Self:
        self.arr += other
        return self

    def __isub__(self, other: Any) -> Self:
        self.arr -= other
        return self

    def __imul__(self, other: Any) -> Self:
        self.arr *= other
        return self

    def __itruediv__(self, other: Any) -> Self:
        self.arr /= other
        return self

    def __ifloordiv__(self, other: Any) -> Self:
        self.arr //= other
        return self

    def __imod__(self, other: Any) -> Self:
        self.arr %= other
        return self

    def __ipow__(self, other: Any) -> Self:
        self.arr **= other
        return self

    def __imatmul__(self, other: Any) -> Self:
        self.arr @= other
        return self


class FixedLengthArray(NumericMixins):
    """
    Array to support objects like Coordinate sets or vectors which have "len()" which maps to their number of items
    """
    def __init__(self, arr: ArrayT, **kwargs: dict[str, Any]):
        super().__init__(arr=arr, **kwargs)

    def __iter__(self) -> Generator[tuple[str, Any], None, None]:
        for i in self.arr:
            yield i

    def create_mask(self, selection: IndexLike) -> NDArray[np.bool_] | NDArray[np.int_]:
        """Creates a boolean vector mask that corresponds to row indices"""

        # Case 1: slice object
        if isinstance(selection, slice):
            vector_mask = convert_slice_to_integer_range(selection=selection, length=len(self))

        # Case 2: single integer
        elif isinstance(selection, int):
            vector_mask = np.array([selection])

        # Case 3: numpy arrays and sequences
        else:
            if isinstance(selection, np.ndarray):
                selection = cast(np.ndarray, np.atleast_1d(selection.squeeze()))
            vector_mask = Vector_IndexT(selection)

        # Case 3a: Boolean
        if vector_mask.dtype == np.bool_:
            if vector_mask.shape[0] != len(self):
                raise ValueError(f"Mask has wrong number of points. Mask:{vector_mask.size}  != array:{len(self)}")
            return vector_mask

        # Case 3b: Integer
        else:
            mask = np.zeros(len(self), dtype=np.bool_)
            mask[vector_mask] = True

            if np.sum(mask) < len(vector_mask):
                logger.warning(f"Oversampling of points in sample, reduce or extract is not supported. "
                               f"{len(vector_mask) - np.sum(mask)} points were oversampled\n"
                               f"Duplicate points are not created.")

            return mask

    def sample(self, index: IndexLike) -> Self:
        """Return a sample of the array"""
        mask = self.create_mask(index)
        return self.copy(self.arr[mask], deep=True)

    def reduce(self, index: IndexLike) -> None:
        """Reduces the array to the points indexed"""
        mask = self.create_mask(index)
        self.arr = self.arr[mask]

    def extract(self, index: IndexLike) -> Self:
        """Returns the points indexed but also reduces the indexed array by these points"""
        mask = self.create_mask(index)
        extracted = self.sample(mask)
        self.reduce(~mask)
        return extracted


class BaseVector(FixedLengthArray):
    arr: VectorT

    def __init__(self, arr: VectorT, **kwargs: dict[str, Any]) -> None:
        super().__init__(arr=arr, **kwargs)

    # noinspection PyNestedDecorators
    @field_validator('arr', mode='before')
    @classmethod
    def _coerce_array(cls, value: ArrayT | Self) -> VectorT:
        value = super(BaseVector, cls)._coerce_array(value)
        return np.atleast_1d(value.squeeze())


class HomogeneousArray(FixedLengthArray):
    def __init__(self, arr: ArrayT, **kwargs: dict[str, Any]) -> None:
        super().__init__(arr, **kwargs)

    # noinspection PyPep8Naming
    @property
    def H(self) -> ArrayT:
        return np.column_stack((self.arr, np.ones(len(self), dtype=self.dtype)))


class ArrayNx2(HomogeneousArray):
    arr: Array_Nx2_T

    def __init__(self, arr: Array_Nx2_T, **kwargs: dict[str, Any]) -> None:
        super().__init__(arr=arr, **kwargs)

    # noinspection PyNestedDecorators
    @field_validator('arr', mode='plain')
    @classmethod
    def _coerce_array(cls, value: ArrayT) -> Array_Nx2_T:
        value = super(ArrayNx2, cls)._coerce_array(value)
        return validate_transposed_2d_array(value, 2)


class ArrayNx3(HomogeneousArray):
    arr: Array_Nx3_Float_T

    def __init__(self, arr: Array_Nx3_Float_T, **kwargs: dict[str, Any]) -> None:
        super().__init__(arr=arr, **kwargs)

    @field_validator('arr', mode='plain')
    @classmethod
    def _coerce_array(cls, value: ArrayT) -> Array_Nx3_Float_T:
        value = super(ArrayNx3, cls)._coerce_array(value)
        return validate_transposed_2d_array(value, 3)

