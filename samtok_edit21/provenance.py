"""Content-addressed cache contract. Paths locate files; hashes identify them."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .data import file_hash, row_hash
from .protocol import validate_row

FORMAT = "samtok21-cache-v2"
PREPROCESSING = "qwen21-samtok-inline-v2-prenorm4096-rgba-resize32"


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
    # Do not hash the unused official TE or the SAM2/codec used only for masks.
    return {
        "dit": tree_identity(Path(qwen) / "transformer"),
        "vae": tree_identity(Path(qwen) / "vae"),
        "scheduler": tree_identity(Path(qwen) / "scheduler", weights=False),
        "te": tree_identity(samtok),
        "processor": tree_identity(Path(qwen) / "processor", weights=False),
    }


def conditioning_identity(qwen, samtok, adapter, max_pixels, metadata=None):
    from .training import adapter_identity
    identity = {
        "schema": FORMAT, "preprocessing": PREPROCESSING,
        "models": model_identity(qwen, samtok),
        "te_adapter": normalize_adapter_identity(adapter_identity(adapter)),
        "max_pixels": max_pixels,
    }
    if metadata is not None:
        from .data import read_rows
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


def normalize_conditioning_identity(identity):
    if identity.get("schema") == FORMAT:
        return identity
    adapter = identity.get("te_adapter_identity", identity.get("te_adapter"))
    return {**identity, "te_adapter": normalize_adapter_identity(adapter)}


def assert_models_match(identity, qwen, samtok):
    if identity.get("schema") != FORMAT or identity.get("preprocessing") != PREPROCESSING:
        raise ValueError("Legacy/unknown conditioning identity; audit and rebuild cache as v2")
    if identity["models"] != model_identity(qwen, samtok):
        raise ValueError("Base model / tokenizer / processor content differs from cache")


def assert_inference_identity(identity, qwen, samtok, adapter):
    from .training import adapter_identity
    identity = normalize_conditioning_identity(identity)
    assert_models_match(identity, qwen, samtok)
    actual = normalize_adapter_identity(adapter_identity(adapter))
    if identity["te_adapter"] != actual:
        raise ValueError("TE adapter weights/config differ from Stage 2 conditioning identity")


def cache_path(directory, relative):
    root = Path(directory).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Cache path must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Cache shard escapes cache directory")
    return path


def verify_cache(directory, manifest):
    from .training import validate_conditioning
    if manifest.get("format") != FORMAT:
        raise ValueError("Stage 2 requires samtok21-cache-v2; legacy cache needs audit/rebuild")
    identity = manifest.get("identity", {})
    if identity.get("schema") != FORMAT or identity.get("preprocessing") != PREPROCESSING:
        raise ValueError("Unknown cache conditioning/preprocessing protocol")
    if not identity.get("models") or not identity.get("metadata_sha256"):
        raise ValueError("Cache is missing source provenance")
    rows = manifest.get("rows", [])
    if not rows or manifest.get("row_count") != len(rows):
        raise ValueError("Missing cache rows")
    ordered_hashes = [row_hash({k: v for k, v in row.items() if not k.startswith("_cache")})
                      for row in rows]
    if "rows_sha256" in identity:
        if identity["rows_sha256"] != row_hash(ordered_hashes):
            raise ValueError("Manifest rows disagree with ordered source row digest")
    elif identity.get("row_hashes") != ordered_hashes:
        # Read compatibility for intermediate v2 smoke artifacts produced
        # before the compact digest optimization; new writers never use this.
        raise ValueError("Cache is missing source row identities")
    seen_indices, seen_paths = set(), set()
    for row in rows:
        original = {k: v for k, v in row.items() if not k.startswith("_cache")}
        validate_row(original)
        if original["sample_type"] == "edit_ntp":
            raise ValueError("Stage 2 cache cannot contain edit_ntp")
        path = cache_path(directory, row.get("_cache_file"))
        if path in seen_paths:
            raise ValueError("Duplicate cache shard")
        seen_paths.add(path)
        side = json.loads(path.with_suffix(".json").read_text())
        index = side.get("row_index")
        if not isinstance(index, int) or index in seen_indices:
            raise ValueError("Duplicate/invalid cache row index")
        seen_indices.add(index)
        if not 0 <= index < len(rows) or ordered_hashes[index] != row_hash(original):
            raise ValueError("Cache row index does not match source metadata")
        if side.get("identity") != identity or side.get("row_hash") != row_hash(original):
            raise ValueError("Mixed or stale cache identity/row metadata")
        if side.get("sha256") != file_hash(path):
            raise ValueError("Cache checksum mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload.get("row_index") != index or payload.get("row_hash") != side["row_hash"]
                or payload.get("identity") != identity):
            raise ValueError("Cache payload provenance disagrees with sidecar")
        inputs = payload["inputs"]
        if "prompt_embeds_mask" not in inputs:
            raise ValueError("Missing text attention mask")
        sources = original["edit_image"]
        source_count = 1 if isinstance(sources, str) else len(sources)
        if len(inputs["edit_latents"]) != source_count:
            raise ValueError("Source latent count differs from metadata image count")
        validate_conditioning(inputs)
    if seen_indices != set(range(len(rows))):
        raise ValueError("Cache row indices are not a complete unique range")
    return True


def audit_legacy_cache(directory, manifest):
    """Read-only audit of both historical layouts; never authorizes training."""
    from .training import verify_cache as verify_old
    if manifest.get("format") != "samtok21-cache-v1":
        raise ValueError("Not a legacy v1 cache")
    if all(Path(row["_cache_file"]).name == row["_cache_file"] for row in manifest["rows"]):
        verify_old(directory, manifest)
    else:
        for row in manifest["rows"]:
            path = cache_path(directory, row["_cache_file"])
            side = json.loads(path.with_suffix(".json").read_text())
            original = {k: v for k, v in row.items() if not k.startswith("_cache")}
            validate_row(original)
            if side.get("sha256") != file_hash(path) or side.get("row_hash") != row_hash(original):
                raise ValueError("Legacy cache checksum/row mismatch")
    return {"checksums_valid": True, "training_eligible": False,
            "reason": "v1 did not record verifiable complete model/config provenance; rebuild v2"}
