"""The docs quote source code; the quotes must not drift from the code."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_sync():
    spec = importlib.util.spec_from_file_location("sync_code_blocks", ROOT / "scripts/docs/sync_code_blocks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_code_blocks_in_docs_match_the_source(capsys):
    sync = load_sync()
    status = sync.main(["--check"])
    report = capsys.readouterr().out
    assert status == 0, "Run `python scripts/docs/sync_code_blocks.py` and commit the docs:\n" + report
    assert " code blocks" in report and not report.startswith("0 code blocks")


def test_excerpt_selects_functions_methods_and_files():
    sync = load_sync()
    method = sync.excerpt("src/samtok_edit21/models/code_injector.py", "CodeInjector.code_index")
    assert method.startswith("def code_index(self, input_ids):") and "return torch.where" in method
    function = sync.excerpt("src/samtok_edit21/training/routing.py", "route_targets")
    assert function.startswith("def route_targets(")
    assert sync.excerpt("scripts/grounding/run_arnold.sh", None).startswith("#!/usr/bin/env bash")


def test_composed_blocks_join_settings_and_entry_without_a_second_shebang():
    sync = load_sync()
    text = sync.compose("scripts/round2/settings/c2_4n.sh + scripts/round2/bootstrap_arnold.sh")
    assert text.startswith("#!/usr/bin/env bash") and text.count("#!/usr/bin/env bash") == 1
    assert text.index("export SAMTOK_RUN_ID=") < text.index("set -Eeuo pipefail")
