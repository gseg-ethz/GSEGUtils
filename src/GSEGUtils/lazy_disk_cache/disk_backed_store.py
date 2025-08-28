import logging
from pathlib import Path
import pickle
import tempfile
from typing import MutableMapping, Optional, cast, Iterator, Any, runtime_checkable, Protocol, TypeGuard, Callable, Unpack

from pydantic import validate_call, ConfigDict
import numpy as np
from numpy.typing import NDArray

# from .disk_backed_ndarray import DiskBackedNDArray
from .lazy_disk_cache import LazyDiskCacheConfig, LazyDiskCache, LazyDiskCacheKw


logger = logging.getLogger(__name__)

# type Array = _NDArray[np.generic]

# @runtime_checkable
# class SupportsOffload(Protocol):
#     def offload(self) -> None: ...

# type Factory[T: LazyDiskCache.rst] = Callable[[_NDArray, Unpack[LazyDiskCacheKw]], T]
@runtime_checkable
class Factory[T: LazyDiskCache](Protocol):
    def __call__(self, data: NDArray, **kwargs: Unpack[LazyDiskCacheKw]) -> T: ...

type Validator[T] = Callable[[object], TypeGuard[T]]


class DiskBackedStore[T: LazyDiskCache](MutableMapping[str, T]):
    _DBNDArrayFileExt = ".pkl"
    
    _store: dict[str, Optional[T]]
    _cache_dir: Path
    _enable_caching: bool
    _automatic_offloading: bool
    _purge_disk_on_gc: bool

    _factory: Factory[T]
    _value_type: Optional[type[T] | tuple[type[T], ...]]
    _validator: Optional[Validator[T]]
    
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
            self,
            *,
            config: LazyDiskCacheConfig = LazyDiskCacheConfig(),
            factory: Factory[T],
            value_type: Optional[type[T] | tuple[type[T], ...]] = None,
            validator: Optional[Validator[T]] = None,
    ) -> None:

        self._store = {}
        self._enable_caching = config.enable_caching
        if config.cache_path is None or config.cache_path.is_file():
            self._cache_dir = Path(tempfile.mkdtemp())
        else:
            self._cache_dir = config.cache_path
        self._automatic_offloading = config.automatic_offloading and config.cache_path is not None
        self._purge_disk_on_gc = config.purge_disk_on_gc

        self._factory = factory
        self._value_type = value_type
        self._validator = validator

        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

            # Scan for existing files
            available_files = [f for f in self._cache_dir.glob(f"*{self._DBNDArrayFileExt}") if f.is_file()]
            for f in available_files:
                self._store[f.stem] = None

    def _check_T(self, value: object) -> T:
        if not isinstance(value, LazyDiskCache):
            raise TypeError(f"value must be LazyDiskCache; got {type(value)}")

        if self._value_type is not None and not isinstance(value, self._value_type):
            raise TypeError(f"value must be {self._value_type}; got {type(value)}")

        if self._validator is not None and not self._validator(value):
            raise TypeError(f"value rejected by validator; got {type(value)}")

        return cast(T, value)

    def __getitem__(self, key: str) -> T:
        obj = self._store.get(key, None)
        if obj is not None:
            return obj

        try:
            with open(self._get_pickle_path(key), "rb") as f:
                loaded_obj = cast(T, pickle.load(f))
        except FileNotFoundError as e:
            raise KeyError(key) from e
        

        self._store[key] = loaded_obj
        return loaded_obj


    def __setitem__(self, key: str, value: T) -> None:
        self._store[key] = self._check_T(value)

    def __delitem__(self, key: str) -> None:
        del self._store[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)
    
    def __contains__(self, key):
        return self._store.__contains__(key)

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"<DiskBackedStore({list(self._store.keys())})>"

    def _get_pickle_path(self, feature: str) -> Path:
        return self._cache_dir / f"{feature}{self._DBNDArrayFileExt}"

    def add_data_to_store(
            self,
            key: str,
            data: NDArray,
            *,
            enable_caching_override: Optional[bool] = None,
            automatic_offloading_override: Optional[bool] = None,
            purge_disk_on_gc_override: Optional[bool] = None,
    ) -> None:
        if key in self:
            raise KeyError(f"Key {key} already exists in store.")
        
        enable_caching = enable_caching_override if enable_caching_override is not None else self._enable_caching
        cache_path=self._cache_dir / f"{key}{self._DBNDArrayFileExt}" if self._cache_dir else None
        automatic_offloading=automatic_offloading_override if automatic_offloading_override is not None else self._automatic_offloading
        purge_disk_on_gc=purge_disk_on_gc_override if purge_disk_on_gc_override is not None else self._purge_disk_on_gc

        new_container = self._factory(
            data,
            enable_caching=enable_caching,
            cache_path=cache_path,
            automatic_offloading=automatic_offloading,
            purge_disk_on_gc=purge_disk_on_gc
        )

        self._store[key] = self._check_T(new_container)

    @property
    def store(self) -> dict[str, Optional[T]]:
        """Returns the internal store dictionary."""
        return self._store

    @property
    def cache_dir(self) -> Path:
        """Returns the directory where cached files are stored."""
        return self._cache_dir


    def keys(self) -> list[str]:
        """
        Returns a list of all available image keys.
        """
        return list(self._store.keys())

    def values(self) -> Iterator[Optional[T]]:
        return iter(self._store.values())

    def items(self) -> Iterator[tuple[str, Optional[T]]]:
        return iter(self._store.items())

    def offload(self, keys: Optional[str | list[str]] = None, pickle_container: bool = False) -> None:
        """ 
        Offloads the specified images from memory to disk. If no keys are provided, all images are offloaded.
        If `pickle_container` is True, the entire DiskBackedNDArray object is pickled to disk instead of just offloading its data.
        """
        if keys is None:
            keys = self.keys()
        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            obj = self._store[key]
            if obj is None:
                continue
            if pickle_container:
                with open(self._get_pickle_path(key), "wb") as f:
                    pickle.dump(obj, f)
                logger.debug(f"Pickled DiskBackedNDArray for {key=} to {self._get_pickle_path(key)}")
            else:
                obj.offload()


    def __getstate__(self) -> dict[str, Any]:
        if self._enable_caching:
            self.offload(pickle_container=True)
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if self._enable_caching:
            for key in list(self.keys()):
                if self._store[key] is not None:
                    continue
                try:
                    with open(self._get_pickle_path(key), "rb") as f:
                        loaded = pickle.load(f)
                    self._store[key] = self._check_T(loaded)
                except FileNotFoundError:
                    logger.warning(f"File for key {key} not found in cache directory {self._cache_dir}.")
                    continue
