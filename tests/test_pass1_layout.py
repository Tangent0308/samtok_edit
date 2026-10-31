"""Regression checks for reorganized imports and CLI argument forwarding."""

import importlib
import sys
from types import SimpleNamespace

import pytest

from samtok_edit21.pass1 import cli


def test_help_does_not_import_a_model_backend(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Top-level help must not import optional model dependencies"
        )

    monkeypatch.setattr(cli.importlib, "import_module", forbidden)
    assert cli.main(["--help"]) == 0
    assert "prepare-curriculum" in capsys.readouterr().out


def test_dispatch_preserves_child_help_and_restores_argv_after_failure(monkeypatch):
    original = sys.argv
    seen = []

    def child():
        seen.append(sys.argv[:])
        raise RuntimeError("child failure")

    def resolve(name):
        assert name == "samtok_edit21.pass1.training.formal"
        return SimpleNamespace(main=child)

    monkeypatch.setattr(cli.importlib, "import_module", resolve)
    with pytest.raises(RuntimeError, match="child failure"):
        cli.main(["train", "--help"])
    assert seen == [["samtok-pass1 train", "--help"]]
    assert sys.argv is original


@pytest.mark.parametrize(
    "old,new,symbol",
    [
        ("formal_train", "training.formal", "rank_samples"),
        ("formal_prepare", "data.curriculum", "pack_batches"),
        ("formal_materialize", "data.materialization", "sample_point"),
        ("score", "evaluation.scoring", "decode_stable"),
        ("train", "training.checkpoint", "save_adapter"),
    ],
)
def test_existing_experiment_imports_resolve_to_canonical_implementation(
    old, new, symbol
):
    legacy = importlib.import_module("samtok_edit21.pass1." + old)
    canonical = importlib.import_module("samtok_edit21.pass1." + new)
    assert getattr(legacy, symbol) is getattr(canonical, symbol)
