"""Compatibility entry point; use ``samtok_edit21.pass1.data.selection``."""

from .data.selection import (
    MODES as MODES,
)
from .data.selection import (
    Selector as Selector,
)
from .data.selection import (
    add_counterfactual_pairs as add_counterfactual_pairs,
)
from .data.selection import (
    area_bin as area_bin,
)
from .data.selection import (
    build_bank as build_bank,
)
from .data.selection import (
    canonicalize_teachers as canonicalize_teachers,
)
from .data.selection import (
    describe as describe,
)
from .data.selection import (
    distinct as distinct,
)
from .data.selection import (
    dump as dump,
)
from .data.selection import (
    eligible_modes as eligible_modes,
)
from .data.selection import (
    holdout_requests as holdout_requests,
)
from .data.selection import (
    main as main,
)
from .data.selection import (
    rebalance_modes as rebalance_modes,
)
from .data.selection import (
    region_key as region_key,
)
from .data.selection import (
    split_sources as split_sources,
)
from .data.selection import (
    stable as stable,
)

if __name__ == "__main__":
    main()
