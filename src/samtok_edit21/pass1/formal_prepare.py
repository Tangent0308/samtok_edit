"""Compatibility entry point; use ``samtok_edit21.pass1.data.curriculum``."""

from .data.curriculum import (
    context_length as context_length,
)
from .data.curriculum import (
    main as main,
)
from .data.curriculum import (
    pack_batches as pack_batches,
)

if __name__ == "__main__":
    main()
