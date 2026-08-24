#!/usr/bin/env python3
"""Audit that an edit validation split is disjoint from training metadata."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_source_parquet(name: str) -> str:
    return Path(name).stem.replace(" ", "_") + ".parquet"


def source_identity(row: dict) -> tuple[str, int] | None:
    provenance = row.get("provenance") or {}
    source_parquet = provenance.get("source_parquet")
    row_idx = provenance.get("row_idx")
    if source_parquet is not None and row_idx is not None:
        return canonical_source_parquet(str(source_parquet)), int(row_idx)

    edit_image = row.get("edit_image")
    if not isinstance(edit_image, str):
        return None
    path = Path(edit_image)
    row_prefix, separator, _ = path.name.partition("_source.")
    if not separator or not row_prefix.isdigit() or not path.parent.name:
        return None
    return f"{path.parent.name}.parquet", int(row_prefix)


def resolve_image(path: str, base_path: Path) -> Path:
    image_path = Path(path)
    return image_path if image_path.is_absolute() else base_path / image_path


def sha256_file(path: Path) -> tuple[Path, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return path, digest.hexdigest()


def stat_file(path: Path) -> tuple[Path, int | None]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return path, None
    return path, stat.st_size if path.is_file() else None


def write_json_atomic(payload: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_metadata", type=Path, required=True)
    parser.add_argument("--candidate_base", type=Path, required=True)
    parser.add_argument("--reference_metadata", type=Path, required=True)
    parser.add_argument("--reference_base", type=Path, required=True)
    parser.add_argument(
        "--reference_sample_types",
        default="edit_mt,edit",
        help="Comma-separated reference sample types included in the audit",
    )
    parser.add_argument("--io_workers", type=int, default=32)
    parser.add_argument("--report_json", type=Path, required=True)
    args = parser.parse_args()
    if args.io_workers < 1:
        raise ValueError("io_workers must be positive")

    candidate_rows = read_jsonl(args.candidate_metadata)
    allowed_reference_types = {
        name.strip() for name in args.reference_sample_types.split(",") if name.strip()
    }
    reference_rows = [
        row
        for row in read_jsonl(args.reference_metadata)
        if row.get("sample_type") in allowed_reference_types
    ]

    candidate_identities = [source_identity(row) for row in candidate_rows]
    if any(identity is None for identity in candidate_identities):
        raise ValueError("Every candidate row must have a recoverable source identity")
    candidate_identity_set = set(candidate_identities)
    reference_identities_by_type = defaultdict(set)
    for row in reference_rows:
        identity = source_identity(row)
        if identity is None:
            raise ValueError(
                f"Reference {row.get('sample_type')} row lacks a source identity"
            )
        reference_identities_by_type[row["sample_type"]].add(identity)

    identity_overlap_by_type = {
        sample_type: len(candidate_identity_set & identities)
        for sample_type, identities in sorted(reference_identities_by_type.items())
    }

    def refs(rows, key):
        return {row[key] for row in rows if isinstance(row.get(key), str)}

    candidate_source_refs = refs(candidate_rows, "edit_image")
    candidate_target_refs = refs(candidate_rows, "image")
    reference_source_refs = refs(reference_rows, "edit_image")
    reference_target_refs = refs(reference_rows, "image")
    relative_ref_overlap = {
        "source_to_source": len(candidate_source_refs & reference_source_refs),
        "target_to_target": len(candidate_target_refs & reference_target_refs),
        "source_to_any": len(
            candidate_source_refs & (reference_source_refs | reference_target_refs)
        ),
        "target_to_any": len(
            candidate_target_refs & (reference_source_refs | reference_target_refs)
        ),
    }

    candidate_paths_by_role = {
        "source": {
            resolve_image(path, args.candidate_base) for path in candidate_source_refs
        },
        "target": {
            resolve_image(path, args.candidate_base) for path in candidate_target_refs
        },
    }
    reference_paths_by_role = {
        "source": {
            resolve_image(path, args.reference_base) for path in reference_source_refs
        },
        "target": {
            resolve_image(path, args.reference_base) for path in reference_target_refs
        },
    }
    all_candidate_paths = set().union(*candidate_paths_by_role.values())
    all_reference_paths = set().union(*reference_paths_by_role.values())
    all_audit_paths = sorted(all_candidate_paths | all_reference_paths)
    with concurrent.futures.ThreadPoolExecutor(args.io_workers) as executor:
        path_sizes = dict(executor.map(stat_file, all_audit_paths))
    missing = [str(path) for path, size in path_sizes.items() if size is None]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} audit images are missing; first={missing[0]}"
        )

    with concurrent.futures.ThreadPoolExecutor(args.io_workers) as executor:
        candidate_hash_pairs = list(executor.map(sha256_file, all_candidate_paths))
    candidate_hashes = {path: digest for path, digest in candidate_hash_pairs}
    candidate_sizes = {path_sizes[path] for path in all_candidate_paths}
    # An exact byte duplicate must have the same file size. Stat all reference
    # files, then hash only size-compatible paths to keep a 40k audit cheap.
    size_matched_reference_paths = {
        path for path in all_reference_paths if path_sizes[path] in candidate_sizes
    }
    with concurrent.futures.ThreadPoolExecutor(args.io_workers) as executor:
        reference_hash_pairs = list(
            executor.map(sha256_file, size_matched_reference_paths)
        )
    reference_hashes = {path: digest for path, digest in reference_hash_pairs}

    candidate_all_hashes = set(candidate_hashes.values())
    reference_all_hashes = set(reference_hashes.values())
    content_overlap = {
        "candidate_any_to_reference_any": len(
            candidate_all_hashes & reference_all_hashes
        ),
        "candidate_source_to_reference_any": len(
            {
                candidate_hashes[path]
                for path in candidate_paths_by_role["source"]
            }
            & reference_all_hashes
        ),
        "candidate_target_to_reference_any": len(
            {
                candidate_hashes[path]
                for path in candidate_paths_by_role["target"]
            }
            & reference_all_hashes
        ),
    }
    candidate_internal_duplicate_hashes = len(candidate_hashes) - len(
        candidate_all_hashes
    )

    failures = {
        **{f"identity:{key}": value for key, value in identity_overlap_by_type.items()},
        **{f"relative_ref:{key}": value for key, value in relative_ref_overlap.items()},
        **{f"content:{key}": value for key, value in content_overlap.items()},
        "candidate_internal_duplicate_identities": len(candidate_identities)
        - len(candidate_identity_set),
        "candidate_internal_duplicate_hashes": candidate_internal_duplicate_hashes,
    }
    report = {
        "passed": not any(failures.values()),
        "candidate": {
            "metadata": str(args.candidate_metadata.resolve()),
            "rows": len(candidate_rows),
            "unique_source_identities": len(candidate_identity_set),
            "unique_source_images": len(candidate_source_refs),
            "unique_target_images": len(candidate_target_refs),
        },
        "reference": {
            "metadata": str(args.reference_metadata.resolve()),
            "sample_types": sorted(allowed_reference_types),
            "rows": len(reference_rows),
            "unique_source_identities_by_type": {
                key: len(value)
                for key, value in sorted(reference_identities_by_type.items())
            },
            "unique_source_identities_union": len(
                set().union(*reference_identities_by_type.values())
            ),
            "unique_referenced_images": len(all_reference_paths),
        },
        "identity_overlap_by_type": identity_overlap_by_type,
        "relative_reference_overlap": relative_ref_overlap,
        "content_sha256_overlap": content_overlap,
        "content_hash_audit": {
            "candidate_images_hashed": len(candidate_hashes),
            "reference_images_total": len(all_reference_paths),
            "reference_images_size_matched_and_hashed": len(reference_hashes),
        },
        "candidate_internal_duplicate_identities": failures[
            "candidate_internal_duplicate_identities"
        ],
        "candidate_internal_duplicate_content_hashes": candidate_internal_duplicate_hashes,
        "failures": {key: value for key, value in failures.items() if value},
    }
    write_json_atomic(report, args.report_json)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise ValueError(f"Metadata overlap audit failed: {report['failures']}")


if __name__ == "__main__":
    main()
