#!/usr/bin/env python3
"""Keep the code excerpts in the Markdown docs identical to the source they quote.

A documented excerpt is fenced between two HTML comments (invisible when rendered):

    <!-- code: src/samtok_edit21/models/code_injector.py::CodeInjector.forward -->
    ```python
    ...
    ```
    <!-- /code -->

``python scripts/docs/sync_code_blocks.py`` rewrites every such block in ``docs/*.md`` from the current source: a
function, a ``Class.method``, a class, or the whole file when there is no ``::symbol``.  Several sources joined with
`` + `` are concatenated, a later file losing its ``#!`` line: a paste-ready script is its settings file followed by
the entry it configures, e.g. ``<!-- code: scripts/round2/settings/c2_4n.sh + scripts/round2/bootstrap_arnold.sh -->``.  ``--check`` changes nothing,
lists the blocks that are out of date and exits 1; the test suite runs it, so a code change that forgets the docs
fails the tests.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = r"[^\s:]+(?:::[\w.]+)?"
BLOCK = re.compile(
    rf"(?P<open><!-- code: (?P<spec>{SOURCE}(?: \+ {SOURCE})*) -->\n)"
    r"```(?P<lang>\w*)\n(?P<body>.*?)\n?```\n(?P<close><!-- /code -->)", re.S)


def excerpt(path, symbol):
    source = (ROOT / path).read_text(encoding="utf-8")
    if not symbol:
        return source.rstrip("\n")
    node = ast.parse(source)
    for name in symbol.split("."):
        matches = [child for child in ast.iter_child_nodes(node)
                   if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child.name == name]
        if len(matches) != 1:
            raise LookupError(f"{path}::{symbol}: expected one definition of {name!r}, found {len(matches)}")
        node = matches[0]
    first = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
    lines = source.split("\n")[first - 1:node.end_lineno]
    return textwrap.dedent("\n".join(lines)).rstrip("\n")


def compose(spec):
    """The text a block stands for: one source, or several joined with " + " (later files without their shebang)."""
    parts = []
    for position, source in enumerate(spec.split(" + ")):
        path, _, symbol = source.partition("::")
        body = excerpt(path, symbol or None)
        if position and body.startswith("#!"):
            body = body.split("\n", 1)[1] if "\n" in body else ""
        parts.append(body)
    return "\n".join(parts)


def sync(document, check):
    text = document.read_text(encoding="utf-8")
    stale = []

    def replace(match):
        body = compose(match["spec"])
        if body != match["body"]:
            stale.append(match["spec"])
        return f"{match['open']}```{match['lang']}\n{body}\n```\n{match['close']}"

    updated = BLOCK.sub(replace, text)
    if stale and not check:
        document.write_text(updated, encoding="utf-8")
    return len(BLOCK.findall(text)), stale


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="Report stale blocks and exit 1 instead of rewriting")
    args = parser.parse_args(argv)
    total, failures = 0, 0
    for document in sorted((ROOT / "docs").glob("*.md")):
        count, stale = sync(document, args.check)
        total += count
        for name in stale:
            failures += 1
            print(f"{'stale' if args.check else 'updated'}: {document.name}: {name}")
    print(f"{total} code blocks, {failures} {'stale' if args.check else 'updated'}")
    return 1 if args.check and failures else 0


if __name__ == "__main__":
    sys.exit(main())
