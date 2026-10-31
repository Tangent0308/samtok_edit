"""Compatibility entry point; use ``samtok_edit21.pass1.evaluation.pilot_report``."""

from .evaluation.pilot_report import (
    case_figures as case_figures,
)
from .evaluation.pilot_report import (
    charts as charts,
)
from .evaluation.pilot_report import (
    main as main,
)
from .evaluation.pilot_report import (
    metric as metric,
)
from .evaluation.pilot_report import (
    overlay as overlay,
)
from .evaluation.pilot_report import (
    pct as pct,
)
from .evaluation.pilot_report import (
    read as read,
)
from .evaluation.pilot_report import (
    rows as rows,
)

if __name__ == "__main__":
    main()
