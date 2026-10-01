"""SAMTok adaptation for Qwen-Image-2.1, with lazy public model APIs."""
from . import api

__all__ = api.__all__
__version__ = "0.1.0"


def __getattr__(name):
    return getattr(api, name)
