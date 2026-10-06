import json

import pytest
import torch

from samtok_edit21.data.io import file_hash, row_hash
from samtok_edit21.data.protocol import box_of, span_of
from samtok_edit21.data.provenance import FORMAT, PREPROCESSING, validate_cache_manifest, verify_cache
from samtok_edit21.models.binding import GEOMETRY, SCHEMA

PLAIN = dict(sample_type="edit", edit_type="attribute", edit_image="source.png",
             image="target.png", prompt="Turn the car blue.")
UMT = dict(sample_type="edit_umt", edit_type="add", edit_image="source.png", image="target.png",
           instr_variant="noref", prompt="Add a red ball in this region " + box_of((0, 0, 500, 500)) + ".")


def conditioning(region=False):
    # 2x2 source/target grids: one TE image pad maps to four latent tokens.
    inputs = {
        "input_latents": torch.zeros(1, 64, 2, 2),
        "edit_latents": [torch.zeros(1, 64, 2, 2)],
        "prompt_embeds": torch.zeros(1, 8, 4096),
        "prompt_embeds_mask": torch.ones(1, 8, dtype=torch.long),
        "edit_image_pad_mask": torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool),
    }
    if region:
        maps = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
        inputs["region_binding"] = {"schema": SCHEMA, "geometry": GEOMETRY, "units": [[4, 7]],
                                    "instruction": [3, 8], "length": 8, "empty": [False],
                                    "target": maps.clone(), "source": maps.clone()}
    return inputs


def write_cache(root, rows):
    hashes = [row_hash(row) for row in rows]
    identity = {"schema": FORMAT, "preprocessing": PREPROCESSING, "models": {"fixture": "synthetic"},
                "te_adapter": None, "binding": {"schema": SCHEMA}, "max_pixels": 65536,
                "metadata_sha256": "fixture", "rows_sha256": row_hash(hashes)}
    manifest_rows = []
    for index, row in enumerate(rows):
        shard = root / "0" / f"{index}.pth"
        shard.parent.mkdir(exist_ok=True)
        torch.save({"inputs": conditioning(row["sample_type"] == "edit_umt"), "row_index": index,
                    "row_hash": hashes[index], "identity": identity}, shard)
        shard.with_suffix(".json").write_text(json.dumps(
            {"row_index": index, "identity": identity, "row_hash": hashes[index], "sha256": file_hash(shard)}))
        manifest_rows.append({**row, "_cache_file": f"0/{index}.pth"})
    return {"format": FORMAT, "identity": identity, "row_count": len(rows), "rows": manifest_rows}


def test_cache_identity_binding_payloads_and_checksums(tmp_path):
    manifest = write_cache(tmp_path, [PLAIN, UMT])
    assert verify_cache(tmp_path, manifest)
    stale = json.loads(json.dumps(manifest))
    stale["rows"][0]["prompt"] = "Turn the car red."
    with pytest.raises(ValueError, match="digest"):
        verify_cache(tmp_path, stale)
    adapter = json.loads(json.dumps(manifest))
    adapter["identity"]["te_adapter"] = {"sha256": "x"}
    with pytest.raises(ValueError, match="raw TE"):
        validate_cache_manifest(adapter)
    payload = torch.load(tmp_path / "0/1.pth", weights_only=True)
    payload["inputs"].pop("region_binding")
    torch.save(payload, tmp_path / "0/1.pth")
    side = json.loads((tmp_path / "0/1.json").read_text())
    side["sha256"] = file_hash(tmp_path / "0/1.pth")
    (tmp_path / "0/1.json").write_text(json.dumps(side))
    with pytest.raises(ValueError, match="binding payload"):
        verify_cache(tmp_path, manifest)
    (tmp_path / "0/0.pth").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verify_cache(tmp_path, manifest)


def test_binding_payload_geometry_is_checked():
    from samtok_edit21.training.objectives import validate_conditioning

    inputs = conditioning(region=True)
    validate_conditioning(inputs)
    for change in (dict(units=[[0, 3]]), dict(units=[[4, 9]]), dict(empty=[True]),
                   dict(target=torch.ones(1, 3, 2)), dict(target=torch.full((1, 2, 2), 2.0))):
        broken = conditioning(region=True)
        broken["region_binding"].update(change)
        with pytest.raises(ValueError):
            validate_conditioning(broken)


def test_geometry_requires_qwen3_vae_alignment():
    from samtok_edit21.training.objectives import validate_conditioning

    inputs = {
        "input_latents": torch.zeros(1, 64, 4, 4),
        "prompt_embeds": torch.zeros(1, 8, 4096),
        "edit_image_pad_mask": torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool
        ),
        "edit_latents": [torch.zeros(1, 64, 4, 4)],
    }
    validate_conditioning(inputs)
    inputs["edit_image_pad_mask"][0, 0] = False
    with pytest.raises(ValueError, match="disagree"):
        validate_conditioning(inputs)


def test_mask_rows_use_region_payload_rule():
    from samtok_edit21.data.provenance import validate_cache_inputs

    row = {**UMT, "edit_type": "remove", "prompt": "Remove the object in this region " + span_of([3, 300]) + "."}
    validate_cache_inputs(conditioning(region=True), row)
    with pytest.raises(ValueError):
        validate_cache_inputs(conditioning(region=True), PLAIN)
