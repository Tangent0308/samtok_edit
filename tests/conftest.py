import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIFFSYNTH = ROOT / "DiffSynth-Studio"
for path in (ROOT, DIFFSYNTH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
