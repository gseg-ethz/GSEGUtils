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

"""NDArray-shaped object backed by a pickle file on disk.

Implements :class:`DiskBackedNDArray`, a thin wrapper that combines
:class:`LazyDiskCache` (offload-on-pressure semantics) with
:class:`numpy.lib.mixins.NDArrayOperatorsMixin` (transparent participation in
NumPy ufuncs) so callers can use the cache entry as if it were an ``ndarray``.
"""

from typing import Unpack

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.typing import DTypeLike, NDArray

from .lazy_disk_cache import LazyDiskCache, LazyDiskCacheKw


class DiskBackedNDArray(LazyDiskCache, NDArrayOperatorsMixin):
    """Disk-backed view of a single ``numpy.ndarray``.

    Combines the offload-on-pressure semantics of :class:`LazyDiskCache` with the
    NDArray operator surface of :class:`numpy.lib.mixins.NDArrayOperatorsMixin`,
    so callers can index into the cache entry or pass it to a ufunc as if it were
    a regular ``ndarray``. The underlying buffer is materialised on demand
    whenever an attribute marked with :func:`LazyDiskCache.ensure_loaded` is read.

    Parameters
    ----------
    data : NDArray
        The initial in-memory array. Its shape and dtype are cached so the cache
        entry can be described while the buffer is offloaded.
    **settings : Unpack[LazyDiskCacheKw]
        Forwarded to :class:`LazyDiskCache.__init__`.
    """

    def __init__(self, data: NDArray, **settings: Unpack[LazyDiskCacheKw]) -> None:
        """Initialize with ``data`` in memory and forward cache settings to the base class."""
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
        """Return a NumPy view/copy of the underlying buffer.

        Parameters
        ----------
        dtype : numpy.typing.DTypeLike, optional
            Target dtype; when ``None`` the source dtype is preserved.
        copy : bool, optional
            Must be ``None`` or ``True``. ``False`` is rejected because this
            object always materialises through a fresh allocation.

        Returns
        -------
        numpy.ndarray
            A freshly-allocated array with the requested dtype.

        Raises
        ------
        ValueError
            If ``copy=False`` is requested.
        """
        if copy is False:
            raise ValueError("`copy=False` isn't supported. A copy is always created.")

        arr = self._data
        return arr.astype(dtype, copy=True) if dtype else arr.copy()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Reject ufunc participation; tracked as BUG-01 (see CONCERNS.md)."""
        raise NotImplementedError("Not implemented.")

    @LazyDiskCache.ensure_loaded
    def __getitem__(self, key):
        """Return ``self._data[key]`` after loading from disk if necessary."""
        return self._data[key]

    @property
    def data(self):
        """Return the underlying ndarray, loading it from disk on demand."""
        if self.offloaded:
            self.load()
        return self._data

    def _describe_buffer(self) -> tuple[tuple[int, ...], DTypeLike, np.ndarray]:
        return self._shape, self._dtype, self._data

    def _drop_buffer(self) -> None:
        self._data = None  # type: ignore

    def _describe_shape_dtype(self) -> tuple[tuple[int, ...], DTypeLike]:
        return self._shape, self._dtype

    def _set_buffer(self, buf: NDArray) -> None:
        self._data = buf
