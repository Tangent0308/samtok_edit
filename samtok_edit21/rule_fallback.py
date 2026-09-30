"""Compatibility alias for :mod:`samtok_edit21.annotation.rule_fallback`."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("samtok_edit21.annotation.rule_fallback")
_sys.modules[__name__] = _implementation
