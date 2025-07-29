from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
import threading
from typing import Optional, Any, Literal, Unpack, TypedDict, Required, NotRequired
import logging
import weakref
import tempfile
import os

import numpy as np
from numpy.typing import NDArray, DTypeLike

from pydantic import validate_call, ConfigDict
from pydantic.dataclasses import dataclass

from GSEGUtils.config import get_defaults, CacheDefaults

logger = logging.getLogger(__name__)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True), frozen=True)
class LazyDiskCacheConfig:
    enable_caching: bool = False
    cache_path: Optional[Path] = None
    purge_disk_on_gc: bool = True
    automatic_offloading: bool = False

class LazyDiskCacheKw(TypedDict, total=False):
    enable_caching: bool
    cache_path: Optional[Path]
    purge_disk_on_gc: bool
    automatic_offloading: bool

class LazyDiskCache(ABC):
    _MEMMAP_SUFFIX = ".dat"

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
            self,
            **settings: Unpack[LazyDiskCacheKw],
            # **overrides: Unpack[CacheDefaults],
    ) -> None:
        config = LazyDiskCacheConfig(**settings)
        self._init_from_config(config)


    def _init_from_config(self, config: LazyDiskCacheConfig) -> None:
        # defaults = get_defaults()
        # automatic_offloading = overrides.get("preset_automatic_offloading", defaults["preset_automatic_offloading"])
        self._enable_caching = config.enable_caching
        if config.cache_path is None:
            fd, cache_path = tempfile.mkstemp(suffix=self._MEMMAP_SUFFIX) # Todo: Think about a case where the provided path is a dir
            os.close(fd)
            self._cache_path = Path(cache_path)
        else:
            self._cache_path = config.cache_path
        # elif cache_path.is_dir():

        # self._cache_path = cache_path.with_suffix(self._MEMMAP_SUFFIX) if cache_path else None
        self._automatic_offloading = config.automatic_offloading #and (cache_path is not None)
        self._purge_disk_on_gc = config.purge_disk_on_gc

        self._lock = threading.RLock()
        # track in-memory state
        self._in_memory = True
        self._mmap: Optional[np.memmap] = None
        if self._enable_caching:
            self._convert_to_memmap()
        else:
            self._convert_to_ndarray()

        if config.automatic_offloading:
            self.offload()
        if self._cache_path and self._purge_disk_on_gc:
            self._finalizer = weakref.finalize(self, lambda p = self._cache_path: p.unlink(missing_ok=True))


    def _convert_to_memmap(self) -> None:
        """
        Allocate (or reopen) the persistent memmap file and switch
        your live buffer to it, but do *not* flush or drop the old array.
        """
        with self._lock:
            shape, dtype, array = self._describe_buffer()
            array = np.array(array, dtype=dtype, copy=True)  # fully fault into RAM

            # 1) allocate or reopen the mmap file
            if self._mmap is None:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                mode: Literal["w+", "r+"] = "w+" if not self._cache_path.exists() else "r+"
                self._mmap = np.memmap(self._cache_path, dtype=dtype, mode=mode, shape=shape)
            elif self._mmap.mode != "r+":
                self._mmap = np.memmap(self._cache_path, dtype=dtype, mode="r+", shape=shape)

            self._mmap[:] = array

            # 2) hand the memmap off to your subclass as the live buffer
            self._set_buffer(self._mmap)
            self._in_memory = True
            logger.debug(f"Switched buffer to memmap @ {self._cache_path}")

    def _convert_to_ndarray(self) -> None:
        """Pull the memmap entirely into RAM as a plain ndarray."""
        with self._lock:
            # if there's no mmap on disk yet, nothing to do
            if self._mmap is None or not self._cache_path.exists():
                return

            # load (or reload) the mmap so we can read it
            shape, dtype = self._describe_shape_dtype()
            mem = np.memmap(self._cache_path, dtype=dtype, mode="r", shape=shape)

            # copy it into a true ndarray
            array = np.array(mem, dtype=dtype, copy=True)

            # hand it back to the subclass
            self._set_buffer(array)
            self._in_memory = True

            # close & drop the mmap handle
            try:
                # there's no explicit close(), but deleting the object frees it
                del self._mmap
            finally:
                self._mmap = None
                logger.debug(f"Converted memmap @ {self._cache_path} to ndarray")

    @staticmethod
    def ensure_loaded(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            logger.debug(f"Ensure_loaded was called for {func.__name__}")
            was_offloaded = self.offloaded
            if was_offloaded:
                self.load()
            result = func(self, *args, **kwargs)
            if self.automatic_offloading and was_offloaded:
                self.offload()
            return result
        return wrapper

    @property
    def offloaded(self) -> bool:
        return not self._in_memory

    @property
    def automatic_offloading(self) -> bool:
        return self._automatic_offloading

    @automatic_offloading.setter
    def automatic_offloading(self, value: bool):
        assert isinstance(value, bool)
        self._automatic_offloading = value

    @property
    def cache_enabled(self) -> bool:
        return self._enable_caching

    def enable_caching(self) -> None:
        if self._enable_caching:
            return
        self._enable_caching = True
        self._convert_to_memmap()
        logger.debug(f"Enabled caching for {self._cache_path}")

        if self._automatic_offloading:
            self.offload()

    def disable_caching(self) -> None:
        if not self._enable_caching:
            return
        self.load()
        self._enable_caching = False
        self._convert_to_ndarray()
        logger.debug(f"Disabled caching for {self._cache_path}")

    @property
    def purge_disk_on_gc(self) -> bool:
        return self._purge_disk_on_gc

    def disable_purge(self):
        """
        Disable automatic deletion of the cache file when the object is garbage-collected.
        """
        with self._lock:
            if hasattr(self, '_finalizer'):
                self._finalizer.detach()
            self._purge_disk_on_gc = False
            logger.debug(f"Disabled purge for {self._cache_path}")


    def enable_purge(self):
        """
        Enable automatic deletion of the cache file when the object is garbage-collected.
        Registers a new finalizer if needed.
        """
        with self._lock:
            if not self._cache_path:
                return
            if hasattr(self, '_finalizer') and self._finalizer.alive:
                # already enabled
                self._purge_disk_on_gc = True
                return
            # register a new finalizer
            self._finalizer = weakref.finalize(self, lambda p=self._cache_path: p.unlink(missing_ok=True))
            self._purge_disk_on_gc = True
            logger.debug(f"Enabled purge for {self._cache_path}")

    @property
    def cache_path(self) -> Optional[Path]:
        return self._cache_path

    @cache_path.setter
    def cache_path(self, value: Path):
        assert isinstance(value, Path)
        self._cache_path = value
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)

    def offload(self) -> None:
        """Flush the current buffer to disk, drop the in-RAM array, and mark offloaded."""
        with self._lock:
            if not self._enable_caching or self.offloaded:
                return

            # make sure we have a memmap buffer
            if self._mmap is None:
                self._convert_to_memmap()

            # 1) copy any non-memmap array into the memmap
            shape, dtype, array = self._describe_buffer()
            if not isinstance(self._mmap, np.memmap):
                # should not really happen, but just in case:
                self._mmap[:] = np.array(array, dtype=dtype, copy=True)
            else:
                # if subclass buffer is plain ndarray, copy it in
                if not isinstance(array, np.memmap):
                    self._mmap[:] = array

            # 2) flush to disk
            self._mmap.flush()

            # 3) drop the Python buffer and memmap handle
            self._drop_buffer()
            del self._mmap
            self._mmap = None
            self._in_memory = False

            try:
                self.on_offload()
            except Exception:
                logger.exception("Error in on_offload hook")
            logger.debug("Flushed buffer to disk")


    def load(self, mode: Literal["r", "r+", "w+", "c"] = "r+") -> None:
        """Reload the buffer from disk into memory (i.e. make self._mmap your active buffer)."""
        with self._lock:
            if not self.offloaded or self._cache_path is None:
                return

            # (re)open the mmap if needed
            shape, dtype = self._describe_shape_dtype()
            if (self._mmap is None
                    or self._mmap.shape  != shape
                    or self._mmap.dtype  != np.dtype(dtype)
                    or self._mmap.mode   != mode):
                self._mmap = np.memmap(self._cache_path,
                                       dtype=dtype,
                                       mode=mode,
                                       shape=shape)

            # hand that mmap to your subclass as its “buffer”
            self._set_buffer(self._mmap)
            self._in_memory = True
            try:
                self.on_load()
            except Exception:
                logger.exception("Error in on_load hook")
            logger.debug(f"Loaded buffer from {self._cache_path}.\\r\\n"
                         f"Min value: {np.nanmin(self._mmap)}; Max value: {np.nanmax(self._mmap)}")

    def __getstate__(self):
        if hasattr(self, "_finalizer"):
            self._finalizer.detach()

        if self.cache_path:
            self.offload()
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()
        if self._cache_path and self._purge_disk_on_gc:
            self._finalizer = weakref.finalize(self, lambda p=self._cache_path: p.unlink(missing_ok=True))


    @abstractmethod
    def _describe_buffer(self) -> tuple[tuple[int, ...], DTypeLike, NDArray]:
        """Return (shape, dtype, in-memory array)"""
        ...

    @abstractmethod
    def _drop_buffer(self) -> None:
        """Drop the in-memory array (e.g. set to None or similar)"""
        ...

    @abstractmethod
    def _describe_shape_dtype(self) -> tuple[tuple[int, ...], DTypeLike]:
        """Return (shape, dtype) without accessing the full array"""
        ...

    @abstractmethod
    def _set_buffer(self, buf: NDArray) -> None:
        """Given a memmap, restore it into your object"""
        ...

    # Optional hooks subclasses can override
    def on_offload(self) -> None:
        """Hook called after offloading to disk. Use to prune resources."""
        pass

    def on_load(self) -> None:
        """Hook called after loading into memory. Use to cleanup or reinitialize."""
        pass