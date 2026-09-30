"""Compatibility alias for :mod:`samtok_edit21.schema.data`."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("samtok_edit21.schema.data")
_sys.modules[__name__] = _implementation
