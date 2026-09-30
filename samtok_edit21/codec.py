"""Compatibility alias for :mod:`samtok_edit21.models.codec`."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("samtok_edit21.models.codec")
_sys.modules[__name__] = _implementation
