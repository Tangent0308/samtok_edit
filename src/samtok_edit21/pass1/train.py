"""Compatibility entry point; use ``samtok_edit21.pass1.training.pilot``."""

from .training.pilot import (
    Requests as Requests,
)
from .training.pilot import (
    main as main,
)
from .training.pilot import (
    pass2_probe as pass2_probe,
)
from .training.pilot import (
    save_adapter as save_adapter,
)
from .training.pilot import (
    singleton as singleton,
)

if __name__ == "__main__":
    main()
