"""Content-addressed cache contract. Paths locate files; hashes identify them.

v2 conditioning (cache format v3) is produced by the raw SAMTok TE: the
identity records ``te_adapter: None`` plus the codec and binding geometry used
for the region payloads.  The pass-1 localization adapter is not part of the
conditioning identity; inference checks it separately.  Format v2 identities
(v1 runs, Stage-1 TE conditioning) remain readable for inference only.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from samtok_edit21.data.io import file_hash, row_hash
from samtok_edit21.data.protocol import regions_in, validate_row

FORMAT = "samtok21-cache-v3"
PREPROCESSING = "qwen21-samtok-inline-v3-rawte-prenorm4096-rgba-resize32"
LEGACY_FORMAT = "samtok21-cache-v2"
LEGACY_PREPROCESSING = "qwen21-samtok-inline-v2-prenorm4096-rgba-resize32"


def tree_identity(directory, *, weights=True):
    root = Path(directory)
    suffixes = {".json", ".txt", ".jinja", ".model"}
    if weights:
        suffixes |= {".safetensors", ".bin"}
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes
                   and p.name not in {"README.md", "args.json"})
    if not files or (weights and not any(p.suffix in {".safetensors", ".bin"} for p in files)):
        raise ValueError(f"Missing model/config files in {root}")
    return {str(p.relative_to(root)): file_hash(p) for p in files}


def model_identity(qwen, samtok):
    # Do not hash the unused official TE; the codec is recorded with the binding.
    return {
        "dit": tree_identity(Path(qwen) / "transformer"),
        "vae": tree_identity(Path(qwen) / "vae"),
        "scheduler": tree_identity(Path(qwen) / "scheduler", weights=False),
        "te": tree_identity(samtok),
        "processor": tree_identity(Path(qwen) / "processor", weights=False),
    }


def codec_identity(samtok):
    from samtok_edit21.models.binding import GEOMETRY, SCHEMA
    return {"schema": SCHEMA, "geometry": GEOMETRY,
            "codec_sha256": file_hash(Path(samtok) / "mask_tokenizer_256x2.pth"),
            "sam2_sha256": file_hash(Path(samtok) / "sam2.1_hiera_large.pt")}


def conditioning_identity(qwen, samtok, max_pixels, metadata=None):
    """Raw-TE conditioning identity of a v2 Stage 2 cache."""
    identity = {
        "schema": FORMAT, "preprocessing": PREPROCESSING,
        "models": model_identity(qwen, samtok),
        "te_adapter": None,
        "binding": codec_identity(samtok),
        "max_pixels": max_pixels,
    }
    if metadata is not None:
        from samtok_edit21.data.io import read_rows
        identity["metadata_sha256"] = file_hash(metadata)
        # Keep shard identities O(1) in dataset size. Repeating the full list in
        # each payload/sidecar would make metadata storage quadratic.
        identity["rows_sha256"] = row_hash([row_hash(row) for row in read_rows(metadata)])
    return identity


def normalize_adapter_identity(value):
    if value is None:
        return None
    if isinstance(value, str):
        raise ValueError("Legacy path-only adapter identity has no historical hash; rebuild cache")
    if not isinstance(value, dict) or not value.get("sha256"):
        raise ValueError("Missing historical adapter checksum")
    return {k: value[k] for k in ("sha256", "config_sha256") if k in value}


def assert_models_match(identity, qwen, samtok, *, allow_legacy=False):
    known = {(FORMAT, PREPROCESSING)}
    if allow_legacy:
        known.add((LEGACY_FORMAT, LEGACY_PREPROCESSING))
    if (identity.get("schema"), identity.get("preprocessing")) not in known:
        raise ValueError("Unknown conditioning identity; rebuild the cache with v2 code")
    if identity["models"] != model_identity(qwen, samtok):
        raise ValueError("Base model / tokenizer / processor content differs from cache")


def pass2_text_encoder(identity, qwen, samtok, te_adapter):
    """Which TE a Stage 2 DiT adapter expects in pass 2: ``raw`` or ``adapter``.

    v2 adapters were trained on raw-TE conditioning, so pass 2 must disable any
    loaded localization adapter.  v1 adapters (legacy identity) were trained on
    Stage-1-TE conditioning and require exactly that adapter in pass 2.
    """
    from samtok_edit21.training.objectives import adapter_identity
    assert_models_match(identity, qwen, samtok, allow_legacy=True)
    if identity["schema"] == FORMAT:
        if identity.get("te_adapter") is not None:
            raise ValueError("v2 conditioning must come from the raw TE")
        return "raw"
    expected = normalize_adapter_identity(identity.get("te_adapter_identity", identity.get("te_adapter")))
    if expected != normalize_adapter_identity(adapter_identity(te_adapter)):
        raise ValueError("v1 Stage 2 adapters need the exact Stage 1 TE adapter in pass 2")
    return "adapter"


def cache_path(directory, relative):
    root = Path(directory).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Cache path must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Cache shard escapes cache directory")
    return path


def validate_cache_manifest(manifest):
    """Validate the manifest structure without opening cache payloads.

    The full cache can contain hundreds of thousands of files.  This check is
    deliberately limited to metadata, so callers can run payload validation in
    parallel and use this function for the final O(1)-file manifest check.
    """
    if manifest.get("format") != FORMAT:
        raise ValueError(f"Stage 2 requires {FORMAT}; rebuild older caches with v2 code")
    identity = manifest.get("identity", {})
    if identity.get("schema") != FORMAT or identity.get("preprocessing") != PREPROCESSING:
        raise ValueError("Unknown cache conditioning/preprocessing protocol")
    if not identity.get("models") or not identity.get("metadata_sha256") or not identity.get("binding"):
        raise ValueError("Cache is missing source or binding provenance")
    if identity.get("te_adapter") is not None:
        raise ValueError("v2 cache conditioning must come from the raw TE")
    rows = manifest.get("rows", [])
    if not rows or manifest.get("row_count") != len(rows):
        raise ValueError("Missing cache rows")
    ordered_hashes = [row_hash({k: v for k, v in row.items() if not k.startswith("_cache")})
                      for row in rows]
    if identity.get("rows_sha256") != row_hash(ordered_hashes):
        raise ValueError("Manifest rows disagree with ordered source row digest")
    seen_paths = set()
    for row in rows:
        original = {k: v for k, v in row.items() if not k.startswith("_cache")}
        validate_row(original)
        if original["sample_type"] not in {"edit", "edit_umt"}:
            raise ValueError("Stage 2 cache holds only edit/edit_umt rows")
        relative = row.get("_cache_file")
        if not isinstance(relative, str) or not relative:
            raise ValueError("Duplicate or missing cache shard")
        # Reject absolute paths and paths escaping the cache root before any
        # rank-specific worker uses the manifest. Track resolved paths so an
        # equivalent ``a/../b`` spelling cannot duplicate a shard.
        resolved = cache_path(".", relative)
        if resolved in seen_paths:
            raise ValueError("Duplicate cache shard")
        seen_paths.add(resolved)
    return identity, rows, ordered_hashes


def validate_cache_inputs(inputs, row):
    """Payload checks shared by cache publication and Stage 2 startup."""
    from samtok_edit21.training.objectives import validate_conditioning
    if not isinstance(inputs, dict) or "prompt_embeds_mask" not in inputs:
        raise ValueError("Cache payload is missing inputs or the text attention mask")
    sources = row["edit_image"]
    source_count = 1 if isinstance(sources, str) else len(sources)
    if len(inputs["edit_latents"]) != source_count:
        raise ValueError("Source latent count differs from metadata image count")
    if bool(regions_in(row["prompt"])) != (inputs.get("region_binding") is not None):
        raise ValueError("Region rows must carry a binding payload, plain rows none")
    validate_conditioning(inputs)


def _verify_cache_row(directory, row, identity, ordered_hashes, *, side=None):
    """Validate one cache row against its sidecar, checksum and payload."""
    original = {k: v for k, v in row.items() if not k.startswith("_cache")}
    path = cache_path(directory, row.get("_cache_file"))
    if side is None:
        side = json.loads(path.with_suffix(".json").read_text())
    index = side.get("row_index")
    if not isinstance(index, int) or not 0 <= index < len(ordered_hashes):
        raise ValueError("Duplicate/invalid cache row index")
    if ordered_hashes[index] != row_hash(original):
        raise ValueError("Cache row index does not match source metadata")
    if side.get("identity") != identity or side.get("row_hash") != row_hash(original):
        raise ValueError("Mixed or stale cache identity/row metadata")
    if side.get("sha256") != file_hash(path):
        raise ValueError("Cache checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("row_index") != index or payload.get("row_hash") != side["row_hash"]
            or payload.get("identity") != identity):
        raise ValueError("Cache payload provenance disagrees with sidecar")
    validate_cache_inputs(payload["inputs"], original)
    return index


def verify_cache_shard(directory, manifest, rank, world_size):
    """Validate rows ``index % world_size == rank``; any world size can verify any cache."""
    identity, rows, ordered_hashes = validate_cache_manifest(manifest)
    checked = []
    for position in range(rank, len(rows), world_size):
        row = rows[position]
        parts = Path(row["_cache_file"]).parts
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].endswith(".pth"):
            raise ValueError("Cache shard path must be <rank>/<index>.pth")
        index = _verify_cache_row(directory, row, identity, ordered_hashes)
        if index != position:
            raise ValueError("Manifest row order disagrees with cache row indices")
        checked.append(index)
    if not checked:
        raise ValueError(f"Cache manifest has no rows for rank {rank}/{world_size}")
    return checked


def verify_cache(directory, manifest):
    identity, rows, ordered_hashes = validate_cache_manifest(manifest)
    indices = [_verify_cache_row(directory, row, identity, ordered_hashes) for row in rows]
    if sorted(indices) != list(range(len(rows))):
        raise ValueError("Cache row indices are not a complete unique range")
    return True
