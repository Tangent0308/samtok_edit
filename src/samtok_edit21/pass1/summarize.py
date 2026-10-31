"""Compatibility entry point; use ``samtok_edit21.pass1.evaluation.metrics``."""

from .evaluation.metrics import (
    describe as describe,
)
from .evaluation.metrics import (
    main as main,
)
from .evaluation.metrics import (
    paired as paired,
)

if __name__ == "__main__":
    main()
