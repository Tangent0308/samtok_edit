"""Compatibility entry point; use ``samtok_edit21.pass1.data.materialization``."""

from .data.materialization import (
    bounded_subset as bounded_subset,
)
from .data.materialization import (
    conflicting_requests as conflicting_requests,
)
from .data.materialization import (
    decode as decode,
)
from .data.materialization import (
    load_supervision as load_supervision,
)
from .data.materialization import (
    main as main,
)
from .data.materialization import (
    materialize as materialize,
)
from .data.materialization import (
    sample_box as sample_box,
)
from .data.materialization import (
    sample_point as sample_point,
)

if __name__ == "__main__":
    main()
