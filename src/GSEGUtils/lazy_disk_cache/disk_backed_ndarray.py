from pathlib import Path
from typing import Unpack, Self

import numpy as np
from numpy.typing import DTypeLike, NDArray
from numpy.lib.mixins import NDArrayOperatorsMixin

from .lazy_disk_cache import LazyDiskCache, LazyDiskCacheConfig, LazyDiskCacheKw


class DiskBackedNDArray(LazyDiskCache, NDArrayOperatorsMixin):


    def __init__(
            self,
            data: NDArray,
            **settings: Unpack[LazyDiskCacheKw]

    ) -> None:
        self._data = data
        self._shape = data.shape
        self._dtype = data.dtype
        super().__init__(**settings)

    # @classmethod
    # def from_file(
    #         cls,
    #         file_path: str | Path,
    #         **settings: Unpack[LazyDiskCacheKw]
    # ) -> Self:
    #     if not isinstance(file_path, Path):
    #         file_path = str(file_path)
        
    #     if not Path(file_path).exists():
    #         raise FileNotFoundError(f"File '{file_path}' does not exist.")
    #     array_data = np.load(file_path)
    #     return cls(array_data, **settings)

    @LazyDiskCache.ensure_loaded
    def __array__(self, dtype=None, *, copy=None):
        if copy is False:
            raise ValueError("`copy=False` isn't supported. A copy is always created.")

        arr = self._data
        return arr.astype(dtype, copy=True) if dtype else arr.copy()


    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise NotImplementedError("Not implemented.")


    @LazyDiskCache.ensure_loaded
    def __getitem__(self, key):
        return self._data[key]

    @property
    def data(self):
        if self.offloaded:
            self.load()
        return self._data

    def _describe_buffer(self) -> tuple[tuple[int, ...], DTypeLike, np.ndarray]:
        return self._shape, self._dtype, self._data

    def _drop_buffer(self) -> None:
        self._data = None   # type: ignore

    def _describe_shape_dtype(self) -> tuple[tuple[int, ...], DTypeLike]:
        return self._shape, self._dtype

    def _set_buffer(self, buf: NDArray) -> None:
        self._data = buf
