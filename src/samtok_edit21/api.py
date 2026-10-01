"""Lazy public API; importing the package does not load models or CUDA.

Exports point to the original implementation objects, without wrapping model
calls or changing their arguments, return values, or numerical behavior.
"""
from importlib import import_module

_EXPORTS = {
    "load_pipeline": ("samtok_edit21.models.pipeline", "load_pipeline"),
    "edit": ("samtok_edit21.models.pipeline", "edit"),
    "localize": ("samtok_edit21.models.pipeline", "localize"),
    "SamtokCodec": ("samtok_edit21.models.codec", "SamtokCodec"),
    "load_adapter": ("samtok_edit21.training.objectives", "load_adapter"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
