"""Compatibility entry point; use ``samtok_edit21.pass1.evaluation.scoring``."""

from .evaluation.scoring import (
    decode_stable as decode_stable,
)
from .evaluation.scoring import (
    iou as iou,
)
from .evaluation.scoring import (
    main as main,
)

if __name__ == "__main__":
    main()
