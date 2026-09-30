"""Reuse the hash-bound acceptance report from offline data preparation."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .data import file_hash, row_hash


def verify_prepared_region_report(report_path, metadata, region_cache, max_pixels, row_count):
    """Verify small manifests, without reopening image or coverage assets.

    The report is published by full_training_data.merge + full_regions.merge.
    Their workers validated every emitted row/tensor during preparation. Asset
    checks remain in RegionStore.load when training actually consumes a row.
    """
    from ..region.supervision import GEOMETRY, SCHEMA

    report_path = Path(report_path).resolve()
    metadata, region_cache = Path(metadata).resolve(), Path(region_cache).resolve()
    root = report_path.parent
    if metadata != root / "stage1.jsonl" or region_cache != root / "regions":
        raise ValueError("Prepared report must belong to this stage1 metadata and regions directory")
    report = json.loads(report_path.read_text())
    if report.get("training_ready") is not True or report.get("region_cache_ready") is not True:
        raise ValueError("Prepared data report is not training/region ready")
    hashes = {"stage1_sha256": file_hash(metadata),
              "region_manifest_sha256": file_hash(region_cache / "manifest.json")}
    for key, actual in hashes.items():
        if report.get(key) != actual:
            raise ValueError(f"Prepared report {key} differs from current data")
    manifest = json.loads((region_cache / "manifest.json").read_text())
    identity = manifest.get("identity", {})
    if (identity.get("schema") != SCHEMA or identity.get("geometry") != GEOMETRY
            or identity.get("max_pixels") != max_pixels
            or identity.get("metadata_sha256") != hashes["stage1_sha256"]
            or manifest.get("identity_hash") != row_hash(identity)):
        raise ValueError("Prepared region preprocessing identity mismatch")
    records = manifest.get("rows", {})
    if (not isinstance(row_count, int) or row_count <= 0
            or report.get("stage1_rows") != row_count or len(records) != row_count):
        raise ValueError("Prepared metadata/region row counts differ")
    counts = Counter()
    for record in records.values():
        if record.get("eligible") is True:
            if not record.get("coverage_file") or not record.get("sha256"):
                raise ValueError("Prepared eligible region has no coverage provenance")
            counts["eligible"] += 1
        elif record.get("eligible") is False and record.get("reason") in {"task", "empty_region"}:
            counts[record["reason"]] += 1
        else:
            raise ValueError("Unknown prepared region eligibility/reason")
    if dict(counts) != report.get("region_counts"):
        raise ValueError("Prepared region counts differ from report")
    return {"mode": "prepared_report", "report": str(report_path),
            "report_sha256": file_hash(report_path), **hashes,
            "rows": row_count, "region_counts": dict(counts),
            "asset_preflight": "skipped_previously_prepared_data",
            "runtime_checks": "RegionStore.load checks consumed images and coverage"}
