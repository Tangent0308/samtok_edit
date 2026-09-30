"""Compatibility alias for :mod:`samtok_edit21.training_core.calibrate_attention`."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("samtok_edit21.training_core.calibrate_attention")
if __name__ == "__main__":
    _implementation.main()
else:
    _sys.modules[__name__] = _implementation
