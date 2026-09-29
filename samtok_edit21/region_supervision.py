"""Frozen SAMTok regions and area-normalized FM supervision.

Cache coverage is dilated but NOT max-normalized. Attention regions are derived
in FP32 at load time; C always consumes the original coverage union.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .data import file_hash, load_images, read_rows, row_hash, write_json
from .protocol import spans_in

SCHEMA = "samtok-region-supervision-v1"
GEOMETRY = "codec-raw-gt0.5-bilinear-aa-alignfalse-avg16-maxpool3-fp32"


def original_row(row):
    return {k: v for k, v in row.items() if not k.startswith("_")}


def local_umt(row):
    return row["sample_type"] == "edit_umt" and row["edit_type"] != "global"


def coverage_grid(mask, height, width):
    if height % 16 or width % 16:
        raise ValueError("Region canvas must align with VAE tokens")
    pixels = torch.as_tensor(mask, dtype=torch.float32)[None, None]
    pixels = F.interpolate(pixels, (height, width), mode="bilinear",
                           align_corners=False, antialias=True)
    coverage = F.max_pool2d(F.avg_pool2d(pixels, 16, 16), 3, 1, 1)[0, 0]
    # Antialiased interpolation can round a constant-one input slightly above
    # one. Project only the final coverage's numerical roundoff to its range;
    # this is not binary thresholding and preserves all valid cached values.
    return coverage.clamp(0, 1).contiguous()


def attention_regions(coverage):
    coverage = coverage.float()
    peaks = coverage.flatten(1).amax(1)
    if not torch.isfinite(coverage).all() or (peaks <= 0).any():
        raise ValueError("Attention regions must be finite and nonempty")
    return coverage / peaks[:, None, None]


def region_fm_loss(error, coverage, weight=0.5, n_min=16.0):
    """error [B,H,W], union coverage [B,H,W]; average samples independently."""
    if not math.isfinite(weight) or weight < 0 or not math.isfinite(n_min) or n_min <= 0:
        raise ValueError("Invalid region loss coefficients")
    if error.shape != coverage.shape or error.ndim != 3:
        raise ValueError("Region/error geometry mismatch")
    e, m = error.float(), coverage.to(device=error.device, dtype=torch.float32)
    if not torch.isfinite(m).all() or (m < 0).any() or (m > 1).any():
        raise ValueError("Coverage must lie in [0,1]")
    dims = (-2, -1)
    si, so = m.sum(dims), (1 - m).sum(dims)
    di, do = si.clamp_min(n_min), so.clamp_min(n_min)
    inside, outside = (m * e).sum(dims) / di, ((1 - m) * e).sum(dims) / do
    z = 1 + weight * (si / di + so / do)
    loss = ((e.mean(dims) + weight * (inside + outside)) / z).mean()
    position_weights = (1 / (m.shape[-2] * m.shape[-1]) + weight * (
        m / di[:, None, None] + (1 - m) / do[:, None, None])) / z[:, None, None]
    return loss, {"region_inside_mse": inside.mean(), "region_outside_mse": outside.mean(),
                  "region_area": m.mean(), "region_normalizer": z.mean(),
                  "region_inside_tokens": si.mean(), "region_outside_tokens": so.mean(),
                  "region_weight_sum": position_weights.sum(dims).mean(),
                  "region_inside_clamped": (si < n_min).float().mean(),
                  "region_outside_clamped": (so < n_min).float().mean()}


def validate_supervision(value, inputs=None, row=None, *, require_positions=False):
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("Missing/unknown region supervision")
    if type(value.get("eligible")) is not bool or not value.get("identity") or not value.get("row_hash"):
        raise ValueError("Incomplete region identity/eligibility")
    if row is not None:
        row = original_row(row)
        if value["row_hash"] != row_hash(row):
            raise ValueError("Region supervision row mismatch")
        if value["eligible"] and not local_umt(row):
            raise ValueError("Region supervision enabled for an ineligible task")
        if local_umt(row) and not value["eligible"] and value.get("reason") not in {"unaligned", "empty_region"}:
            raise ValueError("Local UMT supervision cannot silently disappear")
    if not value["eligible"]:
        if not value.get("reason"):
            raise ValueError("Skipped supervision requires a reason")
        return
    codes = value.get("spans", [])
    if not codes or any(spans_in(s) != [s] for s in codes):
        raise ValueError("Invalid region spans")
    if row is not None and codes != spans_in(row["prompt"]):
        raise ValueError("Region span order differs from prompt")
    for key in ("coverage_source", "coverage_target"):
        m = value.get(key)
        if not isinstance(m, torch.Tensor) or m.dtype != torch.float32 or m.ndim != 3 or m.shape[0] != len(codes):
            raise ValueError("Invalid FP32 coverage tensor")
        if m.requires_grad or not torch.isfinite(m).all() or (m < 0).any() or (m > 1).any() or (m.flatten(1).amax(1) <= 0).any():
            raise ValueError("Nonfinite, empty or out-of-range region coverage")
    if inputs is not None:
        if len(inputs["edit_latents"]) != 1:
            raise ValueError("Masked supervision requires one source")
        for key, latent in (("coverage_source", inputs["edit_latents"][0]),
                            ("coverage_target", inputs["input_latents"])):
            if value[key].shape[1:] != latent.shape[-2:]:
                raise ValueError("Region/latent grid mismatch")
    positions = value.get("span_positions")
    if require_positions or positions is not None:
        if not isinstance(positions, torch.Tensor) or positions.dtype != torch.long or positions.shape != (len(codes), 4):
            raise ValueError("Missing/invalid mask token positions")
        if (positions < 0).any() or not torch.all(positions[:, 1:] == positions[:, :-1] + 1):
            raise ValueError("Mask token positions must be consecutive")
        if len(codes) > 1 and not torch.all(positions[1:, 0] > positions[:-1, -1]):
            raise ValueError("Mask token positions overlap or are reordered")
        if inputs is not None:
            image_mask = inputs["edit_image_pad_mask"]
            p = positions.to(image_mask.device)
            if int(p.max()) >= image_mask.shape[1] or image_mask[0, p].any():
                raise ValueError("Mask span is outside text positions")
            valid = inputs.get("prompt_embeds_mask")
            if valid is not None and not valid[0, p].bool().all():
                raise ValueError("Mask span lies in text padding")


class RegionStore:
    """Read-only region cache. Verify source files before using training data."""

    def __init__(self, directory, max_pixels):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        identity = self.manifest["identity"]
        if identity.get("schema") != SCHEMA or identity.get("geometry") != GEOMETRY or identity.get("max_pixels") != max_pixels:
            raise ValueError("Region preprocessing identity mismatch")
        self.identity = row_hash(identity)
        if self.manifest.get("identity_hash") != self.identity:
            raise ValueError("Region identity hash mismatch")
        self._verified_files = set()

    def load(self, row, base_path):
        row = original_row(row)
        key = row_hash(row)
        record = self.manifest["rows"].get(key)
        if record is None:
            raise ValueError("Missing row in region cache")
        if record.get("reason") == "task" and record.get("eligible") is False and "file" not in record:
            value = {"schema": SCHEMA, "identity": self.identity, "row_hash": key,
                     "eligible": False, "reason": "task"}
            validate_supervision(value, row=row)
            return value
        from .provenance import cache_path
        path = cache_path(self.directory, record["file"])
        if file_hash(path) != record["sha256"]:
            raise ValueError("Region cache checksum mismatch")
        value = torch.load(path, map_location="cpu", weights_only=True)
        if value.get("identity") != self.identity:
            raise ValueError("Mixed region cache identity")
        if any(value.get(k) != record.get(k) for k in ("eligible", "reason")):
            raise ValueError("Region manifest eligibility mismatch")
        validate_supervision(value, row=row)
        for name, digest in value.get("images", {}).items():
            source = Path(base_path) / name
            signature = (str(source.resolve()), digest)
            if signature not in self._verified_files:
                if file_hash(source) != digest:
                    raise ValueError("Region source/target image content changed")
                self._verified_files.add(signature)
        return value


def main(argv=None):
    from .model import DEFAULT_QWEN, DEFAULT_SAMTOK, build_processor, resize_sources
    from .codec import SamtokCodec

    parser = argparse.ArgumentParser(description="Prepare frozen region supervision before either training stage")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-path", default=".")
    parser.add_argument("--qwen", default=DEFAULT_QWEN)
    parser.add_argument("--samtok", default=DEFAULT_SAMTOK)
    parser.add_argument("--mask-tokenizer-sha256", required=True, help="Checksum of the codec that originally encoded these tokens")
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--device", default="cuda")
    alignment = parser.add_mutually_exclusive_group(required=True)
    alignment.add_argument("--assume-aligned", action="store_true", help="Explicitly certify full-frame source/target alignment for all rows")
    alignment.add_argument("--alignment-manifest", help="JSON mapping row hashes to aligned booleans")
    parser.add_argument("--skip-empty", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Region output must be fresh")
    rows = read_rows(args.metadata)
    tokenizer_path = Path(args.samtok) / "mask_tokenizer_256x2.pth"
    sam_path = Path(args.samtok) / "sam2.1_hiera_large.pt"
    checksum = file_hash(tokenizer_path)
    if checksum != args.mask_tokenizer_sha256:
        raise ValueError("Encoding/decoding codec checksum mismatch")
    alignment_map = json.loads(Path(args.alignment_manifest).read_text()) if args.alignment_manifest else None
    processor = build_processor(args.qwen, args.samtok)
    from diffsynth.pipelines.qwen_image_21 import QwenImage21Unit_EditImageEmbedder
    min_pixels = QwenImage21Unit_EditImageEmbedder.get_processor_min_pixels(SimpleNamespace(processor=processor))
    identity = {"schema": SCHEMA, "geometry": GEOMETRY, "max_pixels": args.max_pixels,
                "codec": checksum, "sam2": file_hash(sam_path),
                "metadata_sha256": file_hash(args.metadata), "skip_empty": args.skip_empty,
                "alignment": "certified-full-frame" if args.assume_aligned else alignment_map,
                "source_min_pixels": min_pixels}
    digest = row_hash(identity)
    codec = SamtokCodec(sam_path, tokenizer_path, device=args.device)
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for row in rows:
        key = row_hash(row)
        if key in records:
            continue
        eligible, reason = local_umt(row), "task"
        if eligible:
            aligned = True if args.assume_aligned else alignment_map.get(key)
            if type(aligned) is not bool:
                raise ValueError("Alignment manifest must explicitly cover every local UMT row")
            eligible, reason = aligned, "unaligned"
        value = {"schema": SCHEMA, "identity": digest, "row_hash": key,
                 "eligible": eligible, "reason": "" if eligible else reason}
        if eligible:
            sources, target, height, width = load_images(row, args.base_path, args.max_pixels)
            if len(sources) != 1:
                raise ValueError("Region supervision requires one source")
            resized = resize_sources(SimpleNamespace(processor=processor), sources, height, width)[0]
            masks = codec.decode_strict(sources[0], row["prompt"])
            spans = spans_in(row["prompt"])
            if len(masks) != len(spans):
                raise ValueError("Decoded mask count mismatch")
            source = torch.stack([coverage_grid(m, resized.height, resized.width) for m in masks])
            target = torch.stack([coverage_grid(m, height, width) for m in masks])
            if (source.flatten(1).amax(1) <= 0).any() or (target.flatten(1).amax(1) <= 0).any():
                if not args.skip_empty:
                    raise ValueError(f"Empty decoded region in {key}")
                value.update(eligible=False, reason="empty_region")
            else:
                value.update(spans=spans, coverage_source=source, coverage_target=target)
            names = row["edit_image"]
            names = [names] if isinstance(names, str) else names
            value["images"] = {name: file_hash(Path(args.base_path) / name) for name in [*names, row["image"]]}
        validate_supervision(value, row=row)
        path = output / (key + ".pt")
        temporary = path.with_suffix(".tmp")
        torch.save(value, temporary)
        temporary.replace(path)
        records[key] = {"file": path.name, "sha256": file_hash(path),
                        "eligible": value["eligible"], "reason": value["reason"]}
    write_json(output / "manifest.json", {"identity": identity, "identity_hash": digest, "rows": records})
    print(json.dumps({"region_rows": len(records), "eligible": sum(r["eligible"] for r in records.values()), "identity": digest}))


if __name__ == "__main__":
    main()
