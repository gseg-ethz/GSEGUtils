from __future__ import annotations

import copy
from abc import ABC
from typing import Any, Generator, MutableMapping, Optional, Self

import numpy as np
import numpy.typing as npt
from numpydantic import NDArray, Shape  # type: ignore
from pydantic import BaseModel, ConfigDict, model_validator

from .base_types import (
    Array_Nx2_T,
    Array_Nx3_T,
    Array_NxM_3_T,
    Array_NxM_T,
    ArrayT,
    IndexLike,
    VectorT,
)


def make_ndarray_type(*args: Optional[int | str], dtype: Optional[npt.DTypeLike] = None) -> NDArray[Any, Any]:
    """
    Helper function to generate the numpydantic type for a ndarray.

    Calling 'make_ndarray_type(None, 3, dtype=np.float32)' would return a numpydantic dtype corresponding to an array
    of shape (N, 3) with dtype = np.float32 and would provide pydantic validation on this
    """
    if len(args) == 0:
        shape_list = ["*", "..."]
    else:
        shape_list = [str(x) if x is not None else "*" for x in args]

    return NDArray[Shape[", ".join(shape_list)], dtype if dtype is not None else Any]


class BaseArray(ABC, BaseModel):
    """
    BaseArray is designed to be a subclassable, automatic validator for array based classes.

    In line with the PCHandler project, the main idea is that it can be extended to support the following:
        -> Coordinate Classes
        -> Scalar Fields
        -> Transformation Matrices (4x4 Affine and 3x3 matrices -> intrinsic / extrinsic)

    The shape of each should be validated any time an object or attribute is changed.
    The dtype should also be able to be validated (e.g. RGB or Optimised Coordinates).
    Objects can also be easily extended to support other attribute information (e.g. lazy loading, image metadata).
    Objects can also be frozen absolutely -> e.g. a loaded read-only array's coordinates "never" change"
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        revalidate_instances="never",
        validate_default=True,
        strict=True,
        frozen=False,
        extra="ignore",
        serialize_by_alias=False,
        populate_by_name=False,
    )
    arr: ArrayT

    @model_validator(mode="after")
    def freeze(self) -> Self:
        if self.model_config["frozen"]:
            self.arr.setflags(write=False)
        return self

    @property
    def __array_interface__(self) -> dict[Any, Any]:
        """Gives access for all numpy functions to the root array object

        All objects will be converted to numpy arrays when processed with numpy functions.
        - __array__ will be deprecated in future -> more reason to use this
        E.g. any function will use np.asarray(base_arraylike.arr.__array_interface__)
        """
        return self.arr.__array_interface__

    @property
    def T(self) -> npt.NDArray[Any]:
        return self.arr.T

    @property
    def shape(self) -> tuple[int, ...]:
        return self.arr.shape

    @property
    def dtype(self) -> npt.DTypeLike[Any]:
        return self.arr.dtype

    @property
    def ndim(self) -> Any:
        return self.arr.ndim

    @property
    def base(self) -> Any:
        return self.arr.base

    @property
    def size(self) -> int:
        return self.arr.size

    def min(self, **kwargs: dict[str, Any]) -> npt.NDArray[Any]:
        return self.arr.min(**kwargs)

    def max(self, **kwargs: dict[str, Any]) -> npt.NDArray[Any]:
        return self.arr.max(**kwargs)

    # def model_dump(self, exclude: set[str]|None = None, **kwargs: dict[str, Any]) -> dict:
    #     """Dumps the model as a serialised dict object"""
    #     exclude = exclude or set()
    #     exclude.add("spher")
    #     return copy.deepcopy(super().model_dump(exclude=exclude))


    def copy(self,
             array: npt.NDArray[Any] | BaseArray | None = None,
             *,
             deep: bool = True,
             update: Optional[MutableMapping[str, Any]] = None,
             **kwargs: dict[str, Any]) -> Self:
        """
        Produce a deep or shallow copy of the model. Updates the model also if parameter is parsed.
        """
        # if not deep:
        #     raise NotImplementedError(f"Shallow copy is not implemented on this class: {type(self)}")

        update = update or {}

        if array is not None:
            if isinstance(array, BaseArray):
                update["arr"] = array.arr
            elif isinstance(array, np.ndarray):
                update["arr"] = array
            else:
                raise TypeError(f'Unknown type passed in for the array of {type(array)}')

        data = self.model_dump(exclude=set(update.keys()), by_alias=False)
        data = copy.deepcopy(data) if deep else data #Todo: Discuss behavior deepcopy should copy the array or not!

        data.update(update)

        return type(self)(**data)

    def view(self, cls: Optional[type] = None) -> Self:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError("Length of an undefined array shape is not clear")

    def __getitem__(self, key: IndexLike) -> npt.NDArray[Any] | Self:
        if isinstance(key, slice):
            key = [key]

        result = self.arr[key]

        if isinstance(result, np.ndarray):
            return self.copy(result)
        return result

    def __setitem__(self, key: IndexLike, value: npt.NDArray[Any] | BaseArray) -> None:
        if isinstance(key, slice):
            key = [key]
        if isinstance(value, BaseArray):
            self.arr[*key] = value.arr
        else:
            self.arr[*key] = value

    def __lt__(self, other: Any) -> npt.NDArray[np.bool_] | bool:
        return self.arr < other

    def __le__(self, other: Any) -> npt.NDArray[np.bool_] | bool:
        return self.arr <= other

    def __eq__(self, other: Any) -> npt.NDArray[np.bool_] | bool:
        return self.arr == other

    def __ne__(self, other: Any) -> npt.NDArray[np.bool_] | bool:
        return self.arr != other

    def __ge__(self, other: Any) -> npt.NDArray[np.bool_] | bool:
        return self.arr >= other

    def __gt__(self, other: Any) -> npt.NDArray[np.bool_] | bool:
        return self.arr > other


class SampleArray(BaseArray):

    def __getitem__(self, key):
        return self.sample(key)

    def create_mask(self, selection: IndexLike, as_vector: bool = False) -> NDArray[np.bool_] | NDArray[np.int_]:
        """Creates a boolean mask for the whole array

        This ensures all new objects are a copy of an array and no views/references
        """
        if isinstance(selection, np.ndarray) and selection.dtype == np.bool_:
            # if selection.ndim > 1: # Todo: Think if this is sensible..in which case do we need to squeeze
            #     selection = selection.squeeze()
            if as_vector:
                if selection.ndim > 1:
                    raise ValueError(f"Selection mask must be a vector like")
                if selection.shape[0] != len(self):
                    raise ValueError(
                        f"Boolean mask does not have the same number of points " f"{selection.size} != {len(self)}"
                    )
                return selection

            if selection.shape == self.shape:
                return selection
            else:
                raise ValueError(f"Invalid selection mask shape: {selection.shape}")

        if as_vector:
            mask = np.zeros(len(self), dtype=np.bool_)
        else:
            mask = np.zeros_like(self.arr, dtype=np.bool_)

        if isinstance(selection, list):
            selection = np.array(selection)

        mask[selection] = True
        return mask

    def sample(self, index: IndexLike) -> Self:
        """Return a sample of the array"""
        mask = self.create_mask(index)
        return self.copy(array=self.arr[mask])

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


class _NumericMixins(BaseArray):
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

    def __divmod__(self, other: Any) -> Self:
        return self.copy(divmod(self.arr, other))

    def __neg__(self, other: Any) -> Self:
        return self.copy(-self.arr)

    def __iadd__(self, other: Any) -> Self:
        self.arr = self.arr + other
        return self

    def __isub__(self, other: Any) -> Self:
        self.arr = self.arr - other
        return self

    def __imul__(self, other: Any) -> Self:
        self.arr = self.arr * other
        return self

    def __itruediv__(self, other: Any) -> Self:
        self.arr = self.arr / other
        return self

    def __ifloordiv__(self, other: Any) -> Self:
        self.arr = self.arr // other
        return self

    def __imod__(self, other: Any) -> Self:
        self.arr = self.arr % other
        return self

    def __ipow__(self, other: Any) -> Self:
        self.arr = self.arr**other
        return self


class FixedLengthArray(SampleArray, _NumericMixins):
    """
    Array to support objects like Coordinate sets or vectors which have "len()" or number of items == rows
    """

    def __len__(self) -> Any:
        return self.arr.shape[0]

    def __iter__(self) -> Generator[tuple[str, Any], None, None]:
        for i in self.arr:
            yield i

    def create_mask(self, selection: IndexLike, **kwargs: dict[str, Any]) -> NDArray[np.bool_] | NDArray[np.int_]:
        return super().create_mask(selection, as_vector=True)

    def sample(self, index: IndexLike) -> Self:
        mask = self.create_mask(index)
        return self.copy(array=self.arr[mask] if self.shape == mask.shape else self.arr[mask, :])

    def reduce(self, index: IndexLike) -> None:
        """Reduces the array to the points indexed"""
        mask = self.create_mask(index)
        self.arr = self.arr[mask] if self.shape == mask.shape else self.arr[mask, :]

    def extract(self, index: IndexLike) -> Self:
        """Returns the points indexed but also reduces the indexed array by these points"""
        mask: npt.NDArray[np.bool_] | npt.NDArray[np.integer] = self.create_mask(index)
        extracted = self.sample(mask)
        self.reduce(~mask)
        return extracted


class BaseVector(FixedLengthArray):
    arr: VectorT


class HomogeneousArray(FixedLengthArray):
    @property
    def H(self) -> npt.NDArray[Any]:
        return np.column_stack((self.arr, np.ones(len(self), dtype=self.dtype)))


class ArrayNx2(HomogeneousArray):
    arr: Array_Nx2_T


class ArrayNx3(HomogeneousArray):
    arr: Array_Nx3_T


class ReadOnlyArray(BaseArray):
    model_config = ConfigDict(strict=True, frozen=True)


class ReadOnlyVector(BaseVector):
    model_config = ConfigDict(strict=True, frozen=True)


class _ImageLike(SampleArray, _NumericMixins, ABC):
    arr: Array_NxM_T | Array_NxM_3_T

    # Update implementation based on if you want to support slicing / views or not
    def __getitem__(self, *key: IndexLike) -> Any:
        return self.arr[*key]

    def create_mask(self, *indices: int|slice):
        if isinstance(indices, slice):
            mask = indices
        else:
            mask = np.zeros_like(self.arr, dtype=np.bool_)
            mask[*indices] = True
        return mask

    def view(self, cls: Optional[type] = None) -> Self:
        return self.copy(self.arr.view(cls=cls), deep=False)


