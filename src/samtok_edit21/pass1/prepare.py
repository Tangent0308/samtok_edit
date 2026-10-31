"""Compatibility entry point; use ``samtok_edit21.pass1.data.pilot``."""

from .data.pilot import (
    MODES as MODES,
)
from .data.pilot import (
    decode_rle as decode_rle,
)
from .data.pilot import (
    dump_json as dump_json,
)
from .data.pilot import (
    geometry as geometry,
)
from .data.pilot import (
    main as main,
)
from .data.pilot import (
    make_request as make_request,
)

if __name__ == "__main__":
    main()
