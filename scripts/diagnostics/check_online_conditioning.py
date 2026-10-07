"""Check on-the-fly Stage 2 conditioning against cached payloads.

Every payload's inputs are recomputed from its metadata row with
``engine.OnlineConditioning`` (the class Stage 2 uses when it has no cache)
and compared tensor by tensor; the row hash and the conditioning identity are
compared as well.  Run on one GPU:

    python scripts/diagnostics/check_online_conditioning.py --metadata <data>/stage2.jsonl \
        --reference <cache rank dir> [<cache rank dir> ...] --output report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from samtok_edit21.data.io import read_rows, row_hash, write_json
from samtok_edit21.data.provenance import conditioning_identity
from samtok_edit21.models.pipeline import DEFAULT_QWEN, DEFAULT_SAMTOK
from samtok_edit21.training.engine import OnlineConditioning, _cpu


def differences(cached, online, path="inputs"):
    """(compared tensors, list of mismatch descriptions) for two payload trees."""
    if isinstance(cached, torch.Tensor):
        if not isinstance(online, torch.Tensor):
            return 1, [f"{path}: tensor vs {type(online).__name__}"]
        if cached.dtype != online.dtype or cached.shape != online.shape:
            return 1, [f"{path}: {cached.dtype}{tuple(cached.shape)} vs {online.dtype}{tuple(online.shape)}"]
        if torch.equal(cached, online):
            return 1, []
        diff = (cached.float() - online.float()).abs().max().item() if cached.is_floating_point() else None
        return 1, [f"{path}: values differ (max abs diff {diff})"]
    if isinstance(cached, dict):
        if not isinstance(online, dict) or set(cached) != set(online):
            return 0, [f"{path}: keys {sorted(cached)} vs {sorted(online) if isinstance(online, dict) else type(online).__name__}"]
        count, issues = 0, []
        for key in sorted(cached):
            n, found = differences(cached[key], online[key], f"{path}.{key}")
            count, issues = count + n, issues + found
        return count, issues
    if isinstance(cached, (list, tuple)):
        if not isinstance(online, (list, tuple)) or len(cached) != len(online):
            return 0, [f"{path}: sequence length/type differs"]
        count, issues = 0, []
        for index, (a, b) in enumerate(zip(cached, online)):
            n, found = differences(a, b, f"{path}[{index}]")
            count, issues = count + n, issues + found
        return count, issues
    return 0, [] if cached == online else [f"{path}: {cached!r} vs {online!r}"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--reference", nargs="+", required=True, help="Directories of cached <index>.pth payloads")
    parser.add_argument("--qwen", default=DEFAULT_QWEN)
    parser.add_argument("--samtok", default=DEFAULT_SAMTOK)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_rows(args.metadata)
    payloads = sorted(p for d in args.reference for p in Path(d).glob("*.pth"))
    first = torch.load(payloads[0], map_location="cpu", weights_only=True)
    max_pixels = first["identity"]["max_pixels"]
    identity = conditioning_identity(args.qwen, args.samtok, max_pixels, args.metadata)
    conditioning = OnlineConditioning(SimpleNamespace(qwen=args.qwen, samtok=args.samtok, device="cuda:0",
                                                      base_path=".", max_pixels=max_pixels))
    results, rng_untouched = [], True
    for path in payloads:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        row = rows[payload["row_index"]]
        cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state(0)
        online = conditioning(row)
        rng_untouched &= torch.equal(cpu_state, torch.get_rng_state()) and torch.equal(cuda_state, torch.cuda.get_rng_state(0))
        devices = {str(t.device) for t in _walk(online)}
        tensors, issues = differences(payload["inputs"], _cpu(online))
        results.append({"payload": str(path), "row_index": payload["row_index"],
                        "kind": row["sample_type"] + ":" + row.get("instr_variant", "") + ":" + row["edit_type"],
                        "region_row": "region_binding" in payload["inputs"],
                        "row_hash_matches": payload["row_hash"] == row_hash(row),
                        "identity_matches": payload["identity"] == identity,
                        "online_devices": sorted(devices), "tensors": tensors, "issues": issues})
        print(json.dumps({"row": payload["row_index"], "tensors": tensors, "issues": len(issues)}), flush=True)
    summary = {"payloads": len(results),
               "bitwise_identical": sum(not r["issues"] for r in results),
               "row_hash_matches": sum(r["row_hash_matches"] for r in results),
               "identity_matches": sum(r["identity_matches"] for r in results),
               "region_rows": sum(r["region_row"] for r in results),
               "tensors_compared": sum(r["tensors"] for r in results),
               "rng_untouched": rng_untouched,
               "kinds": sorted({r["kind"] for r in results})}
    write_json(args.output, {"summary": summary, "results": results})
    print(json.dumps(summary))


def _walk(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)


if __name__ == "__main__":
    main()
