"""Resolve checkout-only assets for cluster launchers, not model inference."""
from pathlib import Path


def repository_root():
    for directory in Path(__file__).resolve().parents:
        if ((directory / "pyproject.toml").is_file()
                and (directory / "scripts/training/run_arnold.sh").is_file()
                and (directory / "third_party/diffsynth").is_dir()):
            return directory
    raise RuntimeError(
        "Cluster orchestration requires an editable repository checkout. "
        "Clone this repository and install requirements.txt; the wheel's "
        "model API and training engine work without checkout scripts."
    )
