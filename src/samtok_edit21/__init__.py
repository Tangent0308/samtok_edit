"""SAMTokEdit for Qwen-Image-2.1: region editing with SAMTok mask tokens, with lazy public model APIs."""
from . import api

__all__ = api.__all__
__version__ = "4.0.0"


def __getattr__(name):
    return getattr(api, name)
