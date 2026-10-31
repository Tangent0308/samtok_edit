"""Compatibility entry point; use ``samtok_edit21.pass1.training.formal``."""

from .training.formal import (
    Requests as Requests,
)
from .training.formal import (
    file_hash as file_hash,
)
from .training.formal import (
    lr_for as lr_for,
)
from .training.formal import (
    main as main,
)
from .training.formal import (
    rank_samples as rank_samples,
)
from .training.formal import (
    restore_rng as restore_rng,
)
from .training.formal import (
    rng_state as rng_state,
)

if __name__ == "__main__":
    main()
