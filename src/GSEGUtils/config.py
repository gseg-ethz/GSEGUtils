from contextvars import ContextVar
from typing import TypedDict, Required, NotRequired, Unpack


class CacheDefaults(TypedDict, total=False):
    preset_automatic_offloading: Required[bool]

_DEFAULTS: ContextVar[CacheDefaults] = ContextVar(
    "_DEFAULTS",
    default=CacheDefaults(preset_automatic_offloading=True),
)

def configure(**defaults: Unpack[CacheDefaults]) -> None:
    current = _DEFAULTS.get()
    current.update(defaults)
    _DEFAULTS.set(current)

def get_defaults() -> CacheDefaults:
    return _DEFAULTS.get()