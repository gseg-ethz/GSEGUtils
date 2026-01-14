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

"""
Provides a class with builtin functionality of:

* Interaction like a `Numpy <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`__ array
* `Pydantic <https://docs.pydantic.dev/latest/>`__ level automatic validation on class and attributes
* Simplistic class and attribute definition like dataclasses

"""

from __future__ import annotations

import copy
import logging
from abc import ABC
from typing import Any, Generator, Optional, Self, TypeVar, cast

import numpy as np
import numpy.typing as npt

# noinspection PyProtectedMember
from numpy._typing._array_like import _ArrayLikeBool_co
from numpydantic import NDArray  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from .base_types import (
    Array_Bool_T,
    Array_Nx2_T,
    Array_Nx3_Float_T,
    ArrayT,
    IndexLike,
    Vector_Bool_T,
    Vector_IndexT,
    VectorT,
)
from .validators import convert_slice_to_integer_range, validate_transposed_2d_array

__all__ = [
    "BaseArray",
    "NumericMixins",
    "FixedLengthArray",
    "BaseVector",
    "HomogeneousArray",
    "ArrayNx2",
    "ArrayNx3",
]

logger = logging.getLogger(__name__)


SelfT = TypeVar("SelfT", bound="BaseArray")


class BaseArray(ABC, BaseModel):
    #: Model config :class:`ConfigDict <pydantic.ConfigDict>`
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Required for numpy types and other newly defined types
        validate_assignment=True,  # Should validate anytime an attribute is set
        revalidate_instances="never",  # Don't keep validating instances (avoids infinite validation loops)
        validate_default=True,  # Ensure default values get validated as well
        strict=True,  # Ensure that no coercion of types occurs - strict typechecking
        frozen=False,  # Object can be manipulated
        extra="ignore",  # Extra fields passed are not stored in the object (e.g., kwargs)
        serialize_by_alias=False,  # Serialization takes the original field names (e.g., 'arr')
        populate_by_name=False,  # Field is not expected to be populated by attribute name if an alias exists
    )

    arr: ArrayT  #: Contains the raw numpy ndarray data

    def __init__(self, arr: ArrayT, **kwargs: dict[str, Any]):
        """Subclassable array supporting all shapes and numeric/boolean dtypes.

        Parameters
        ----------
        arr: ArrayT
            Input array data
        """
        super().__init__(arr=arr, **kwargs)

    # noinspection PyNestedDecorators
    @field_validator("arr", mode="before")
    @classmethod
    def _coerce_array(cls, value: npt.ArrayLike) -> ArrayT:
        """Coerce an object to a numpy array to assign to the new object.

        Parameters
        ----------
        value: npt.ArrayLike

        Returns
        -------
        ArrayT
        """
        # TODO Is this needed? Will PCHandler tests still pass?
        if isinstance(value, BaseArray):
            value = value.arr

        return np.atleast_1d(np.asarray(value))

    @property
    def __array_interface__(self) -> dict[str, Any]:
        """Access to the base __array_interface__ property.

        `__array_interface__` allows interaction with the object from other
        libraries and functions that support the array interface protocol.

        Returns
        -------
        dict[str, Any]
        """
        return self.arr.__array_interface__

    # noinspection PyPep8Naming
    @property
    def T(self) -> Self:
        """Returns a transposed view of the array

        Returns
        -------
        Self
        """
        return self.arr.T

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the array

        Returns
        -------
        tuple[int, ...]
        """
        return self.arr.shape

    @property
    def dtype(self) -> np.dtype:
        """Returns the dtype of the array

        Returns
        -------
        :any:`dtype <numpy.dtype>`

        """
        return self.arr.dtype

    @property
    def ndim(self) -> int:
        """Dimensions in the array

        Returns
        -------
        int
        """
        return self.arr.ndim

    @property
    def base(self) -> ArrayT | None:
        """Returns the base array if the array is a view, otherwise None

        Returns
        -------
        ArrayT | None
        """
        return self.arr.base

    @property
    def size(self) -> int:
        """Returns the number of elements in the array

        Returns
        -------
        int
        """
        return self.arr.size

    def view(self, dtype: np.dtype | None = None, _type: type | None = None) -> ArrayT:
        """Return a view of the underlying array

        Parameters
        ----------
        dtype
        _type

        Returns
        -------
        ArrayT
        """
        dtype = dtype or self.arr.dtype

        _type = _type or type(self.arr)

        return self.arr.view(dtype=dtype, type=_type)

    def min(self, **kwargs: dict[str, Any]) -> Any:
        """Returns `self.arr.min(**kwargs)`
        See `numpy.min <https://numpy.org/doc/2.2/reference/generated/numpy.min.html>`_ for more info.

        Parameters
        ----------
        kwargs: dict[str, Any]

        Returns
        -------
        Any
        """
        return self.arr.min(**kwargs)

    def max(self, **kwargs: dict[str, Any]) -> Any:
        """Returns `self.arr.max(**kwargs)`
        See `numpy.max <https://numpy.org/doc/2.2/reference/generated/numpy.max.html>`_ for more info.

        Parameters
        ----------
        kwargs: dict[str, Any]

        Returns
        -------
        Any
        """
        return self.arr.max(**kwargs)

    def __len__(self) -> int:
        """Returns number of rows in array (self.shape[0])

        Returns
        -------
        int
        """
        return self.shape[0]

    def __getitem__(self, key: IndexLike) -> ArrayT | Self:
        """Get items from the array using numpy style indexing

        Parameters
        ----------
        key: IndexLike

        Returns
        -------
        ArrayT | Self
        """
        try:
            return self.copy(array=self.arr[key], deep=False)
        except ValidationError:
            return self.arr[key]

    def __setitem__(self, key: IndexLike, value: ArrayT | BaseArray) -> None:
        """Sets the value at the given index/indices in the array.

        Parameters
        ----------
        key : IndexLike
        value : ArrayT or BaseArray
        """
        self.arr[key] = np.asarray(value)

    def __lt__(self, other: Any) -> Array_Bool_T:
        """Return self.arr < other"""
        return self.arr < other

    def __le__(self, other: Any) -> Array_Bool_T:
        """Return self.arr <= other"""
        return self.arr <= other

    def __ge__(self, other: Any) -> Array_Bool_T:
        """Return self.arr >= other"""
        return self.arr >= other

    def __gt__(self, other: Any) -> Array_Bool_T:
        """Return self.arr > other"""
        return self.arr > other

    def __eq__(self, other: Any) -> Any:
        """Return self.arr == other"""
        return self.arr == other

    def __ne__(self, other: Any) -> Any:
        """Return self.arr != other"""
        return self.arr != other

    def copy(
        self: Self,  # type: ignore[override]
        array: ArrayT | Self | None = None,
        *,
        deep: bool = True,
        update: Optional[dict[str, Any]] = None,
        **kwargs: dict[str, Any],
    ) -> Self:
        """Creates a copy of this object.

        Also supports deep / shallow copies and overriding update of attributes via the `update` dictionary paramter.

        Parameters
        ----------
        array : ArrayT | Self | None, optional
            If set, directly passed as the `arr` attribute in the new instance
        deep : bool, optional
            Deep/shallow copy flag
        update : dict[str, Any] or None, optional
            Dictionary of attributes to override in the new instance
        kwargs : dict[str, Any]

        Returns
        -------
        Self
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
        """Subclassable array type with Python built-in numerical and logical operators

        Parameters
        ----------
        arr: ArrayT
            Input array data
        """
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
    def __init__(self, arr: ArrayT, **kwargs: dict[str, Any]):
        """Class supporting sample, reduce, extract and mask funcs for row-based data

        E.g., vectors or coordinate sets

        Parameters
        ----------
        arr: ArrayT
            Input array data
        """
        super().__init__(arr=arr, **kwargs)

    # TODO should this generator be different as the typical Generator type definition is different?
    def __iter__(self) -> Generator[tuple[str, Any], None, None]:
        for i in self.arr:
            yield i

    def create_mask(self, selection: IndexLike) -> Vector_Bool_T:
        """Converts a basic or advanced numpy index to a boolean vector mask that corresponds to the row indices

        Parameters
        ----------
        selection: IndexLike

        Returns
        -------
        Vector_Bool_T
        """

        # Case 1: slice object
        if isinstance(selection, slice):
            vector_mask = convert_slice_to_integer_range(
                selection=selection, length=len(self)
            )

        # Case 2: single integer
        elif isinstance(selection, int):
            vector_mask = np.array([selection])

        # Case 3: numpy arrays and sequences
        else:
            if isinstance(selection, np.ndarray):
                selection = np.atleast_1d(selection.squeeze())
            vector_mask = Vector_IndexT(selection)

        # Case 3a: Boolean
        if vector_mask.dtype == np.bool_:
            if vector_mask.shape[0] != len(self):
                raise ValueError(
                    f"Mask has wrong number of points. Mask:{vector_mask.size}  != array:{len(self)}"
                )
            return cast(Vector_Bool_T, vector_mask)

        # Case 3b: Integer
        else:
            mask = np.zeros(len(self), dtype=np.bool_)
            mask[vector_mask] = True

            if np.sum(mask) < len(vector_mask):
                logger.warning(
                    f"Oversampling of points in sample, reduce or extract is not supported. "
                    f"{len(vector_mask) - np.sum(mask)} points were oversampled\n"
                    f"Duplicate points are not created."
                )

            return cast(Vector_Bool_T, mask)

    def sample(self, index: IndexLike) -> Self:
        """Return a sample copy of the array

        Parameters
        ----------
        index: IndexLike

        Returns
        -------
        Vector_Bool_T
        """
        mask = self.create_mask(index)
        return self.copy(self.arr[mask], deep=True)

    def reduce(self, index: IndexLike) -> None:
        """Reduces the array to the points indexed

        Parameters
        ----------
        index: IndexLike

        Returns
        -------

        """
        mask = self.create_mask(index)
        self.arr = self.arr[mask]

    def extract(self, index: IndexLike) -> Self:
        """Splits the array with the indexed points being returned and the object containing the remaining values.

        Parameters
        ----------
        index: IndexLike

        Returns
        -------
        Self
        """
        mask = self.create_mask(index)
        extracted = self.sample(mask)
        self.reduce(~mask)
        return extracted


class BaseVector(FixedLengthArray):
    arr: VectorT

    def __init__(self, arr: VectorT, **kwargs: dict[str, Any]) -> None:
        """Shape validated 1D array

        Parameters
        ----------
        arr: ArrayT
            Input array data
        """
        super().__init__(arr=arr, **kwargs)

    # noinspection PyNestedDecorators
    @field_validator("arr", mode="before")
    @classmethod
    def _coerce_array(cls, value: ArrayT | Self) -> VectorT:
        value = super(BaseVector, cls)._coerce_array(value)
        return np.atleast_1d(value.squeeze())


class HomogeneousArray(FixedLengthArray):
    def __init__(self, arr: ArrayT, **kwargs: dict[str, Any]) -> None:
        """Helper class for homogeneous coordinate creation

        Parameters
        ----------
        arr: ArrayT
            Input array data
        """
        super().__init__(arr, **kwargs)

    # noinspection PyPep8Naming
    @property
    def H(self) -> ArrayT:
        """
        Returns the homogeneous coordinates of the array by adding a column of ones to the right

        Parameters
        ----------

        Returns
        -------
        ArrayT

        """
        return np.column_stack((self.arr, np.ones(len(self), dtype=self.dtype)))


class ArrayNx2(HomogeneousArray):
    """Shape validated Nx2 array

    Parameters
    ----------
    arr: ArrayT
        Input array data
    """

    arr: Array_Nx2_T

    def __init__(self, arr: Array_Nx2_T, **kwargs: dict[str, Any]) -> None:
        super().__init__(arr=arr, **kwargs)

    # noinspection PyNestedDecorators
    @field_validator("arr", mode="plain")
    @classmethod
    def _coerce_array(cls, value: ArrayT) -> Array_Nx2_T:
        value = super(ArrayNx2, cls)._coerce_array(value)
        return validate_transposed_2d_array(value, 2)


class ArrayNx3(HomogeneousArray):
    """Shape validated Nx3 array

    Parameters
    ----------
    arr: ArrayT
        Input array data
    """

    arr: Array_Nx3_Float_T

    def __init__(self, arr: Array_Nx3_Float_T, **kwargs: dict[str, Any]) -> None:
        super().__init__(arr=arr, **kwargs)

    @field_validator("arr", mode="plain")
    @classmethod
    def _coerce_array(cls, value: ArrayT) -> Array_Nx3_Float_T:
        value = super(ArrayNx3, cls)._coerce_array(value)
        return validate_transposed_2d_array(value, 3)
