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

"""Pydantic-validated numpy-array base classes.

Provides :class:`BaseArray` and its subclasses (:class:`NumericMixins`,
:class:`FixedLengthArray`, :class:`BaseVector`, :class:`HomogeneousArray`,
:class:`ArrayNx2`, :class:`ArrayNx3`). Each class behaves like a NumPy array via
the :attr:`__array_interface__` protocol while gaining Pydantic-level field
validation, dataclass-style declarations, and standard arithmetic / comparison
dunders.
"""

from __future__ import annotations

import copy
import logging
from abc import ABC
from typing import Any, Generator, Optional, Self, TypeVar, cast

import numpy as np
import numpy.typing as npt

# noinspection PyProtectedMember
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
    """Abstract Pydantic-validated wrapper around a NumPy array.

    Subclasses inherit the strict validation config and the
    :attr:`__array_interface__` proxy. Concrete subclasses constrain ``arr`` to
    a more specific shape / dtype via :class:`numpydantic.NDArray` annotations
    and (optionally) override :meth:`_coerce_array` to add extra coercion.
    """

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
        """Return a transposed view of the array.

        Returns
        -------
        Self
        """
        return self.arr.T

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the array.

        Returns
        -------
        tuple[int, ...]
        """
        return self.arr.shape

    @property
    def dtype(self) -> np.dtype:
        """Return the dtype of the array.

        Returns
        -------
        :any:`dtype <numpy.dtype>`
        """
        return self.arr.dtype

    @property
    def ndim(self) -> int:
        """Return the number of dimensions in the array.

        Returns
        -------
        int
        """
        return self.arr.ndim

    @property
    def base(self) -> ArrayT | None:
        """Return the base array if this array is a view, otherwise ``None``.

        Returns
        -------
        ArrayT | None
        """
        return self.arr.base

    @property
    def size(self) -> int:
        """Return the number of elements in the array.

        Returns
        -------
        int
        """
        return self.arr.size

    def view(self, dtype: np.dtype | None = None, _type: type | None = None) -> ArrayT:
        """Return a view of the underlying array.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            Target dtype for the view; defaults to the current dtype.
        _type : type, optional
            Sub-type of :class:`numpy.ndarray` to construct; defaults to the
            current array type.

        Returns
        -------
        ArrayT
        """
        dtype = dtype or self.arr.dtype

        _type = _type or type(self.arr)

        return self.arr.view(dtype=dtype, type=_type)

    def min(self, **kwargs: dict[str, Any]) -> Any:
        """Return ``self.arr.min(**kwargs)``.

        See :func:`numpy.min` for the supported keyword arguments.

        Parameters
        ----------
        **kwargs : dict[str, Any]
            Forwarded to :meth:`numpy.ndarray.min`.

        Returns
        -------
        Any
        """
        return self.arr.min(**kwargs)

    def max(self, **kwargs: dict[str, Any]) -> Any:
        """Return ``self.arr.max(**kwargs)``.

        See :func:`numpy.max` for the supported keyword arguments.

        Parameters
        ----------
        **kwargs : dict[str, Any]
            Forwarded to :meth:`numpy.ndarray.max`.

        Returns
        -------
        Any
        """
        return self.arr.max(**kwargs)

    def __len__(self) -> int:
        """Return the number of rows in the array (``self.shape[0]``).

        Returns
        -------
        int
        """
        return self.shape[0]

    def __getitem__(self, key: IndexLike) -> ArrayT | Self:
        """Index into the array using NumPy-style semantics.

        Parameters
        ----------
        key : IndexLike
            Anything accepted by :meth:`numpy.ndarray.__getitem__`.

        Returns
        -------
        ArrayT | Self
            Returns ``Self`` when the slice still validates against this class's
            shape contract; otherwise returns the raw ``ndarray``.
        """
        try:
            return self.copy(array=self.arr[key], deep=False)
        except ValidationError:
            return self.arr[key]

    def __setitem__(self, key: IndexLike, value: ArrayT | BaseArray) -> None:
        """Set the value(s) at ``key`` to ``value``.

        Parameters
        ----------
        key : IndexLike
        value : ArrayT or BaseArray
        """
        self.arr[key] = np.asarray(value)

    def __lt__(self, other: Any) -> Array_Bool_T:
        """Return ``self.arr < other``."""
        return self.arr < other

    def __le__(self, other: Any) -> Array_Bool_T:
        """Return ``self.arr <= other``."""
        return self.arr <= other

    def __ge__(self, other: Any) -> Array_Bool_T:
        """Return ``self.arr >= other``."""
        return self.arr >= other

    def __gt__(self, other: Any) -> Array_Bool_T:
        """Return ``self.arr > other``."""
        return self.arr > other

    def __eq__(self, other: Any) -> Any:
        """Return ``self.arr == other``."""
        return self.arr == other

    def __ne__(self, other: Any) -> Any:
        """Return ``self.arr != other``."""
        return self.arr != other

    def copy(
        self: Self,  # type: ignore[override]
        array: ArrayT | Self | None = None,
        *,
        deep: bool = True,
        update: Optional[dict[str, Any]] = None,
        **kwargs: dict[str, Any],
    ) -> Self:
        """Return a copy of this object.

        Supports deep / shallow copies and per-field overrides via ``update``.

        Parameters
        ----------
        array : ArrayT | Self | None, optional
            If set, directly passed as the ``arr`` attribute in the new instance.
        deep : bool, optional
            Whether to deep-copy non-``arr`` fields. Default ``True``.
        update : dict[str, Any], optional
            Per-field overrides applied to the new instance.
        **kwargs : dict[str, Any]
            Reserved for subclass overrides.

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
    """Adds Python arithmetic / in-place / logical dunders that delegate to ``self.arr``.

    Binary operators return a fresh wrapper of the same type via :meth:`BaseArray.copy`;
    in-place operators mutate ``self.arr`` and return ``self``.
    """

    def __init__(self, arr: ArrayT, **kwargs: dict[str, Any]):
        """Initialize a subclassable numeric/logical array wrapper.

        Parameters
        ----------
        arr : ArrayT
            Input array data.
        """
        super().__init__(arr=arr, **kwargs)

    def __add__(self, other: Any) -> Self:
        """Return ``self + other`` wrapped in a fresh instance."""
        return self.copy(self.arr + other)

    def __sub__(self, other: Any) -> Self:
        """Return ``self - other`` wrapped in a fresh instance."""
        return self.copy(self.arr - other)

    def __mul__(self, other: Any) -> Self:
        """Return ``self * other`` wrapped in a fresh instance."""
        return self.copy(self.arr * other)

    def __truediv__(self, other: Any) -> Self:
        """Return ``self / other`` wrapped in a fresh instance."""
        return self.copy(self.arr / other)

    def __floordiv__(self, other: Any) -> Self:
        """Return ``self // other`` wrapped in a fresh instance."""
        return self.copy(self.arr // other)

    def __mod__(self, other: Any) -> Self:
        """Return ``self % other`` wrapped in a fresh instance."""
        return self.copy(self.arr % other)

    def __pow__(self, other: Any) -> Self:
        """Return ``self ** other`` wrapped in a fresh instance."""
        return self.copy(self.arr**other)

    def __matmul__(self, other: Any) -> Self:
        """Return ``self @ other`` wrapped in a fresh instance."""
        return self.copy(self.arr @ other)

    def __rmatmul__(self, other: Any) -> Self:
        """Return ``other @ self`` wrapped in a fresh instance."""
        return self.copy(other @ self.arr)

    def __radd__(self, other: Any) -> Self:
        """Return ``other + self`` wrapped in a fresh instance."""
        return self.copy(other + self.arr)

    def __rsub__(self, other: Any) -> Self:
        """Return ``other - self`` wrapped in a fresh instance."""
        return self.copy(other - self.arr)

    def __rmul__(self, other: Any) -> Self:
        """Return ``other * self`` wrapped in a fresh instance."""
        return self.copy(other * self.arr)

    def __rpow__(self, other: Any) -> Self:
        """Return ``other ** self`` wrapped in a fresh instance."""
        return self.copy(other**self.arr)

    def __rtruediv__(self, other: Any) -> Self:
        """Return ``other / self`` wrapped in a fresh instance."""
        return self.copy(other / self.arr)

    def __rfloordiv__(self, other: Any) -> Self:
        """Return ``other // self`` wrapped in a fresh instance."""
        return self.copy(other // self.arr)

    def __rmod__(self, other: Any) -> Self:
        """Return ``other % self`` wrapped in a fresh instance."""
        return self.copy(other % self.arr)

    def __divmod__(self, other: npt.ArrayLike) -> tuple[Self, Self]:
        """Return ``(quotient, remainder)`` from :func:`numpy.divmod`, each wrapped."""
        quotient, remainder = np.divmod(self.arr, other)
        return self.copy(quotient), self.copy(remainder)

    def __rdivmod__(self, other: Any) -> tuple[Self, Self]:
        """Return ``(quotient, remainder)`` from ``np.divmod(other, self.arr)``, each wrapped."""
        quotient, remainder = np.divmod(other, self.arr)
        return self.copy(quotient), self.copy(remainder)

    def __neg__(self) -> Self:
        """Return ``-self`` wrapped in a fresh instance."""
        return self.copy(-self.arr)

    def __abs__(self) -> Self:
        """Return ``abs(self)`` wrapped in a fresh instance."""
        return self.copy(np.abs(self.arr))

    def __iadd__(self, other: Any) -> Self:
        """In-place ``self += other``; mutates ``self.arr``."""
        self.arr += other
        return self

    def __isub__(self, other: Any) -> Self:
        """In-place ``self -= other``; mutates ``self.arr``."""
        self.arr -= other
        return self

    def __imul__(self, other: Any) -> Self:
        """In-place ``self *= other``; mutates ``self.arr``."""
        self.arr *= other
        return self

    def __itruediv__(self, other: Any) -> Self:
        """In-place ``self /= other``; mutates ``self.arr``."""
        self.arr /= other
        return self

    def __ifloordiv__(self, other: Any) -> Self:
        """In-place ``self //= other``; mutates ``self.arr``."""
        self.arr //= other
        return self

    def __imod__(self, other: Any) -> Self:
        """In-place ``self %= other``; mutates ``self.arr``."""
        self.arr %= other
        return self

    def __ipow__(self, other: Any) -> Self:
        """In-place ``self **= other``; mutates ``self.arr``."""
        self.arr **= other
        return self

    def __imatmul__(self, other: Any) -> Self:
        """In-place ``self @= other``; mutates ``self.arr``."""
        self.arr @= other
        return self


class FixedLengthArray(NumericMixins):
    """Row-oriented array wrapper with ``sample`` / ``reduce`` / ``extract`` helpers.

    Suitable bases for per-row-typed data such as 3D coordinate sets, RGB triplets,
    or generic vectors-of-vectors.
    """

    def __init__(self, arr: ArrayT, **kwargs: dict[str, Any]):
        """Initialize a row-indexable array wrapper.

        Parameters
        ----------
        arr : ArrayT
            Input array data.
        """
        super().__init__(arr=arr, **kwargs)

    # TODO should this generator be different as the typical Generator type definition is different?
    def __iter__(self) -> Generator[tuple[str, Any], None, None]:
        """Iterate over rows of ``self.arr``."""
        for i in self.arr:
            yield i

    def create_mask(self, selection: IndexLike) -> Vector_Bool_T:
        """Convert a NumPy index into a boolean per-row mask.

        Parameters
        ----------
        selection : IndexLike
            A slice, integer, integer array, or boolean array.

        Returns
        -------
        Vector_Bool_T
            Boolean mask of length ``len(self)``.

        Raises
        ------
        ValueError
            If a boolean mask is provided with the wrong length.
        """
        # Case 1: slice object
        if isinstance(selection, slice):
            vector_mask = convert_slice_to_integer_range(selection=selection, length=len(self))

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
                raise ValueError(f"Mask has wrong number of points. Mask:{vector_mask.size}  != array:{len(self)}")
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
        """Return a fresh wrapper containing only the rows selected by ``index``.

        Parameters
        ----------
        index : IndexLike
            Anything accepted by :meth:`create_mask`.

        Returns
        -------
        Self
            A deep copy holding only the selected rows.
        """
        mask = self.create_mask(index)
        return self.copy(self.arr[mask], deep=True)

    def reduce(self, index: IndexLike) -> None:
        """Mutate ``self.arr`` to keep only the rows selected by ``index``.

        Parameters
        ----------
        index : IndexLike
            Anything accepted by :meth:`create_mask`.
        """
        mask = self.create_mask(index)
        self.arr = self.arr[mask]

    def extract(self, index: IndexLike) -> Self:
        """Split rows: return the selected rows and reduce ``self`` to the rest.

        Parameters
        ----------
        index : IndexLike
            Anything accepted by :meth:`create_mask`.

        Returns
        -------
        Self
            A new wrapper holding the selected rows. ``self`` is mutated in-place
            to keep only the un-selected rows.
        """
        mask = self.create_mask(index)
        extracted = self.sample(mask)
        self.reduce(~mask)
        return extracted


class BaseVector(FixedLengthArray):
    """1-D variant of :class:`FixedLengthArray` with a vector-shape contract on ``arr``."""

    arr: VectorT

    def __init__(self, arr: VectorT, **kwargs: dict[str, Any]) -> None:
        """Initialize a shape-validated 1-D array wrapper.

        Parameters
        ----------
        arr : VectorT
            Input array data.
        """
        super().__init__(arr=arr, **kwargs)

    # noinspection PyNestedDecorators
    @field_validator("arr", mode="before")
    @classmethod
    def _coerce_array(cls, value: ArrayT | Self) -> VectorT:
        value = super(BaseVector, cls)._coerce_array(value)
        return np.atleast_1d(value.squeeze())


class HomogeneousArray(FixedLengthArray):
    """Row-array wrapper that exposes a homogeneous-coordinate view via :attr:`H`."""

    def __init__(self, arr: ArrayT, **kwargs: dict[str, Any]) -> None:
        """Initialize a homogeneous-coordinate-aware row array.

        Parameters
        ----------
        arr : ArrayT
            Input array data.
        """
        super().__init__(arr, **kwargs)

    # noinspection PyPep8Naming
    @property
    def H(self) -> ArrayT:
        """Return the homogeneous-coordinate form of ``self.arr``.

        Returns
        -------
        ArrayT
            ``self.arr`` with an extra all-ones column appended on the right.
        """
        return np.column_stack((self.arr, np.ones(len(self), dtype=self.dtype)))


class ArrayNx2(HomogeneousArray):
    """Shape-validated ``Nx2`` array.

    Parameters
    ----------
    arr : Array_Nx2_T
        Input array data.
    """

    arr: Array_Nx2_T

    def __init__(self, arr: Array_Nx2_T, **kwargs: dict[str, Any]) -> None:
        """Initialize an ``Nx2`` shape-validated array."""
        super().__init__(arr=arr, **kwargs)

    # noinspection PyNestedDecorators
    @field_validator("arr", mode="plain")
    @classmethod
    def _coerce_array(cls, value: ArrayT) -> Array_Nx2_T:
        """Coerce ``value`` to an ``Nx2`` ndarray, transposing when needed."""
        value = super(ArrayNx2, cls)._coerce_array(value)
        return validate_transposed_2d_array(value, 2)


class ArrayNx3(HomogeneousArray):
    """Shape-validated ``Nx3`` array.

    Parameters
    ----------
    arr : Array_Nx3_Float_T
        Input array data.
    """

    arr: Array_Nx3_Float_T

    def __init__(self, arr: Array_Nx3_Float_T, **kwargs: dict[str, Any]) -> None:
        """Initialize an ``Nx3`` shape-validated array."""
        super().__init__(arr=arr, **kwargs)

    @field_validator("arr", mode="plain")
    @classmethod
    def _coerce_array(cls, value: ArrayT) -> Array_Nx3_Float_T:
        """Coerce ``value`` to an ``Nx3`` ndarray, transposing when needed."""
        value = super(ArrayNx3, cls)._coerce_array(value)
        return validate_transposed_2d_array(value, 3)
